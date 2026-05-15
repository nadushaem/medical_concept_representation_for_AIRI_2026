import os
import pickle
import sys
from collections import Counter
from typing import Tuple

import numpy as np
import torch

from models import AVAILABLE_MODELS
from metrics.metric_utils import bootstrap_auroc_ci, bootstrap_auprc_ci

from tfidf_baseline import (
    OUTCOME_CLASSES,
    PARTIAL_INFO_LEVELS,
    prepare_split,
    truncate_trajectory,
)


MODEL_CONFIGS = {
    "word2vec": {
        "ckpt": "logs/full_whole05_shuffle/word2vec_ngram-word/lightning_logs/"
                "yjsee5xv/checkpoints/epoch=54363-step=100000.ckpt",
        "model_class": "word2vec",
        "use_ngrams": False,
    },
    "fasttext": {
        "ckpt": "logs/full_whole05_shuffle/fasttext_ngram-min2-max5/lightning_logs/"
                "yjsee5xv/checkpoints/epoch=5435-step=100000.ckpt",
        "model_class": "fasttext",
        "use_ngrams": True,
    },
    "glove": {
        "ckpt": "logs/full_whole05_shuffle/glove_ngram-word/lightning_logs/"
                "yjsee5xv/checkpoints/epoch=99999-step=300000.ckpt",
        "model_class": "glove",
        "use_ngrams": False,
    },
}

TOKENIZER_DIR = "data/datasets/mimic-iv-2.2/datasets_full/tokenizer"


def load_all_tokenizers() -> dict:
    """Грузит все pickle-файлы токенайзеров из TOKENIZER_DIR."""
    import data  # noqa: F401  для pickle

    tokenizers = {}
    for name in os.listdir(TOKENIZER_DIR):
        path = os.path.join(TOKENIZER_DIR, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            tokenizers[name] = pickle.load(f)
    return tokenizers


def pick_tokenizer_for_model(tokenizers: dict, use_ngrams: bool):
    """Выбирает word-токенайзер (для w2v, glove) или subword (для fastText)."""
    candidates = []
    for name, tok in tokenizers.items():
        has_ngrams = (
            hasattr(tok, "ngram_min_len")
            or hasattr(tok, "ngram_max_len")
        )
        encoder_size = len(getattr(tok, "encoder", {}))
        candidates.append((name, tok, has_ngrams, encoder_size))

    if use_ngrams:
        for name, tok, has_ng, _ in candidates:
            if has_ng:
                print(f"  picked subword tokenizer: {name[:16]}...")
                return tok
        candidates.sort(key=lambda x: -x[3])
        print(f"  picked tokenizer (largest encoder): {candidates[0][0][:16]}...")
        return candidates[0][1]
    else:
        for name, tok, has_ng, _ in candidates:
            if not has_ng:
                print(f"  picked word tokenizer: {name[:16]}...")
                return tok
        candidates.sort(key=lambda x: x[3])
        print(f"  picked tokenizer (smallest encoder): {candidates[0][0][:16]}...")
        return candidates[0][1]


def encode_token(tokenizer, token: str):
    """Кодирует строковый токен в id (int) или список id (list[int] для ngram)."""
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(token)
    return tokenizer.encoder.get(token, tokenizer.encoder.get("[UNK]", 1))


def infer_model_kwargs(state_dict: dict, model_class: str) -> dict:
    """Восстанавливает kwargs конструктора из размеров тензоров чекпоинта."""
    cleaned = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    if model_class == "glove":
        vocab_size, d_embed = cleaned["l_emb.weight"].shape
        return {
            "vocab_sizes": {"total": vocab_size},
            "special_tokens": {"[PAD]": 0},
            "d_embed": d_embed,
        }

    # word2vec / fasttext: общая структура skip gram
    vocab_size, d_embed = cleaned["center_embeddings.weight"].shape
    has_center_fc = "center_fc.weight" in cleaned
    n_neg_samples = 0 if has_center_fc else 5

    vocab_sizes = {"total": vocab_size}
    if model_class == "fasttext":
        vocab_sizes["ngram"] = vocab_size

    return {
        "vocab_sizes": vocab_sizes,
        "special_tokens": {"[PAD]": 0},
        "d_embed": d_embed,
        "n_neg_samples": n_neg_samples,
    }


def load_model(model_name: str, device: str = "cpu") -> torch.nn.Module:
    """Загружает обученный чекпоинт, восстанавливая параметры из тензоров."""
    config = MODEL_CONFIGS[model_name]
    ckpt_path = config["ckpt"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading {model_name} from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    kwargs = infer_model_kwargs(state_dict, config["model_class"])
    print(f"  Inferred kwargs: {kwargs}")

    ModelClass = AVAILABLE_MODELS[config["model_class"]]
    model = ModelClass(**kwargs)

    cleaned = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  WARN: missing keys: {missing}")
    if unexpected:
        print(f"  WARN: unexpected keys: {unexpected}")

    model.eval()
    return model


def build_token_embedding_dict(
    model: torch.nn.Module,
    tokenizer,
    vocab: list[str],
) -> dict[str, np.ndarray]:
    """Считает эмбеддинг каждого токена один раз; пропускает OOV."""
    encodings = []
    valid_tokens = []
    for tok in vocab:
        try:
            enc = encode_token(tokenizer, tok)
            encodings.append(enc)
            valid_tokens.append(tok)
        except (KeyError, AttributeError):
            continue

    with torch.no_grad():
        embeddings = model.get_token_embeddings(encodings).numpy()
    return {tok: emb for tok, emb in zip(valid_tokens, embeddings)}


def compute_token_counts(trajectories: list[list[str]]) -> dict[str, int]:
    """Частоты токенов на train для inverse frequency весов."""
    counter = Counter()
    for traj in trajectories:
        counter.update(traj)
    return dict(counter)


def embed_trajectory(
    tokens: list[str],
    embedding_dict: dict[str, np.ndarray],
    count_dict: dict[str, int],
) -> np.ndarray:
    """Взвешенное среднее эмбеддингов токенов траектории."""
    vectors, weights = [], []
    for tok in tokens:
        if tok in embedding_dict and tok in count_dict:
            vectors.append(embedding_dict[tok])
            weights.append(1.0 / count_dict[tok])

    if not vectors:
        dim = next(iter(embedding_dict.values())).shape[0]
        return np.zeros(dim, dtype=np.float32)

    matrix = np.stack(vectors)
    w = np.asarray(weights, dtype=matrix.dtype)
    return (matrix * w[:, None]).sum(axis=0) / w.sum()


def embed_all_trajectories(
    trajectories: list[list[str]],
    embedding_dict: dict[str, np.ndarray],
    count_dict: dict[str, int],
) -> np.ndarray:
    """Матрица эмбеддингов всех траекторий (n_samples, d_embed)."""
    return np.stack([
        embed_trajectory(t, embedding_dict, count_dict) for t in trajectories
    ])


def cosine_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity между матрицей a (n, d) и вектором b (d,)."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b)
    a_safe = a / np.clip(a_norm, 1e-12, None)
    b_safe = b / max(b_norm, 1e-12)
    return a_safe @ b_safe


def score_outcome_task(
    traj_matrix: np.ndarray,
    embedding_dict: dict[str, np.ndarray],
    neg_label: str,
    pos_label: str,
) -> np.ndarray:
    """Score(traj) = cos(traj, pos_label) - cos(traj, neg_label)."""
    if pos_label not in embedding_dict or neg_label not in embedding_dict:
        raise KeyError(f"Outcome labels {pos_label}/{neg_label} not in vocab")
    pos_emb = embedding_dict[pos_label]
    neg_emb = embedding_dict[neg_label]
    return cosine_score(traj_matrix, pos_emb) - cosine_score(traj_matrix, neg_emb)


def evaluate_task(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 100,
) -> dict:
    """AUROC / AUPRC с bootstrap CI (без BLB — выборка слишком мала)."""
    common = dict(n_bootstraps=n_bootstraps, use_bag_of_little_bootstraps=False)
    auroc_mean, auroc_std, _ = bootstrap_auroc_ci(y_true, y_score, **common)
    auprc_mean, auprc_std, _ = bootstrap_auprc_ci(y_true, y_score, **common)
    return {
        "auroc_mean": auroc_mean, "auroc_std": auroc_std,
        "auprc_mean": auprc_mean, "auprc_std": auprc_std,
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
    }


def paired_bootstrap_auroc_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Парный bootstrap разности AUROC двух моделей на одной подвыборке."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    attempts = 0
    while len(diffs) < n_bootstraps and attempts < n_bootstraps * 10:
        attempts += 1
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_sample, scores_a[idx])
            auc_b = roc_auc_score(y_sample, scores_b[idx])
        except ValueError:
            continue
        diffs.append(auc_a - auc_b)
    diffs = np.asarray(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def run_model_eval(
    model_name: str,
    tokenizers: dict,
    train_trajectories: list[list[str]],
    train_labels: dict[str, np.ndarray],
    test_trajectories: list[list[str]],
    test_labels: dict[str, np.ndarray],
) -> Tuple[dict, dict]:
    """Прогоняет одну модель по всем P и задачам."""
    print(f"\n=== Evaluating {model_name} ===")
    model = load_model(model_name)
    tokenizer = pick_tokenizer_for_model(
        tokenizers, use_ngrams=MODEL_CONFIGS[model_name]["use_ngrams"],
    )

    all_tokens = set()
    for traj in train_trajectories + test_trajectories:
        all_tokens.update(traj)
    for neg, pos in OUTCOME_CLASSES.values():
        all_tokens.update([neg, pos])
    vocab = sorted(all_tokens)

    print(f"  Computing embeddings for {len(vocab)} tokens...")
    embedding_dict = build_token_embedding_dict(model, tokenizer, vocab)
    print(f"  → {len(embedding_dict)} tokens successfully embedded")
    count_dict = compute_token_counts(train_trajectories)

    results_by_p = {}
    scores_by_task_p = {}
    for partial in PARTIAL_INFO_LEVELS:
        results_by_p[partial] = {}
        test_truncated = [truncate_trajectory(t, partial) for t in test_trajectories]
        test_matrix = embed_all_trajectories(test_truncated, embedding_dict, count_dict)

        for task, (neg_label, pos_label) in OUTCOME_CLASSES.items():
            y_test = test_labels[task]
            if len(np.unique(y_test)) < 2:
                results_by_p[partial][task] = {"skipped": "one class in test"}
                continue
            try:
                scores = score_outcome_task(test_matrix, embedding_dict, neg_label, pos_label)
            except KeyError as e:
                results_by_p[partial][task] = {"skipped": str(e)}
                continue
            results_by_p[partial][task] = evaluate_task(y_test, scores)
            scores_by_task_p[(task, partial)] = scores

    return results_by_p, scores_by_task_p


def print_per_model_results(model_name: str, results: dict) -> None:
    """Таблица AUROC/AUPRC одной модели по всем (P, задача)."""
    print(f"\n--- {model_name} ---")
    header = f"{'P':>5} | {'task':<15} | {'AUROC':>14} | {'AUPRC':>14} | {'n+/n-':>10}"
    print(header)
    print("-" * len(header))
    for p in sorted(results):
        for task, res in results[p].items():
            if "skipped" in res:
                print(f"{p:>5.1f} | {task:<15} | {res['skipped']}")
                continue
            auroc = f"{res['auroc_mean']:.3f} ± {res['auroc_std']:.3f}"
            auprc = f"{res['auprc_mean']:.3f} ± {res['auprc_std']:.3f}"
            counts = f"{res['n_pos']}/{res['n_neg']}"
            print(f"{p:>5.1f} | {task:<15} | {auroc:>14} | {auprc:>14} | {counts:>10}")



def main(data_dir: str) -> None:
    """Главный пайплайн."""
    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    print(f"Loading train: {train_path}")
    train_trajectories, train_labels = prepare_split(train_path)
    print(f"  → {len(train_trajectories)} admissions")
    print(f"Loading test: {test_path}")
    test_trajectories, test_labels = prepare_split(test_path)
    print(f"  → {len(test_trajectories)} admissions")

    print("\nLoading tokenizers...")
    tokenizers = load_all_tokenizers()
    print(f"  → {len(tokenizers)} tokenizer files loaded")
    for name, tok in tokenizers.items():
        enc_size = len(getattr(tok, "encoder", {}))
        has_ng = hasattr(tok, "ngram_min_len")
        print(f"     {name[:16]}... encoder_size={enc_size} has_ngrams={has_ng}")

    all_results = {}
    all_scores = {}
    for model_name in MODEL_CONFIGS:
        try:
            results, scores = run_model_eval(
                model_name, tokenizers,
                train_trajectories, train_labels,
                test_trajectories, test_labels,
            )
            all_results[model_name] = results
            all_scores[model_name] = scores
        except Exception as e:
            print(f"\n!!! {model_name} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n=========== RESULTS PER MODEL ===========")
    for model_name, results in all_results.items():
        print_per_model_results(model_name, results)


if __name__ == "__main__":
    default_dir = "./data/datasets/mimic-iv-2.2/datasets_full"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    main(data_dir)
