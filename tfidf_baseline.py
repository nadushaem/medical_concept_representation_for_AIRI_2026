"""
TF-IDF baseline для задач предсказания исходов из статьи Bornet et al. (2025).

Воспроизводит без обучения эмбеддингов три бинарные задачи:
    - mortality:      LBL_ALIVE vs LBL_DEAD
    - readmission:    LBL_AWAY  vs LBL_READM
    - length-of-stay: LBL_SHORT vs LBL_LONG

Каждая траектория представляется TF-IDF вектором над словарём токенов. Для
каждого класса считается центроид по train. На test predicted score =
sim(positive_centroid) - sim(negative_centroid). Метрики AUROC / AUPRC с
доверительными интервалами через bootstrap (n=100), как в статье.

Используется тот же bootstrap из metrics/metric_utils.py авторов, чтобы цифры
были сопоставимы с их выходом. BLB отключён (subsample_size > n_test).
"""

import json
import os
import sys
from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from metrics.metric_utils import bootstrap_auroc_ci, bootstrap_auprc_ci

OUTCOME_CLASSES = {
    "mortality":      ("LBL_ALIVE", "LBL_DEAD"),
    "readmission":    ("LBL_AWAY",  "LBL_READM"),
    "length-of-stay": ("LBL_SHORT", "LBL_LONG"),
}
ALL_LABEL_TOKENS = {tok for pair in OUTCOME_CLASSES.values() for tok in pair}

PARTIAL_INFO_LEVELS = [0.0, 0.1, 0.3, 0.6, 1.0]


def load_split(path: str) -> list[list[str]]:
    """Читает JSON-сплит (по одной траектории на строку) в список списков."""
    trajectories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def extract_labels_and_tokens(
    trajectory: list[str],
) -> Tuple[dict, list[str]]:
    """Отделяет три label-токена от остальной траектории."""
    labels = {}
    clean_tokens = []
    for tok in trajectory:
        if tok in ALL_LABEL_TOKENS:
            for task, (neg, pos) in OUTCOME_CLASSES.items():
                if tok == pos:
                    labels[task] = 1
                elif tok == neg:
                    labels[task] = 0
        elif tok.startswith("SUB_") or tok.startswith("ADM_"):
            continue
        else:
            clean_tokens.append(tok)
    return labels, clean_tokens


def truncate_trajectory(tokens: list[str], partial: float) -> list[str]:
    """Берёт первые partial * 100% не-демографических токенов плюс все DEM."""
    dem_tokens = [t for t in tokens if t.startswith("DEM_")]
    other_tokens = [t for t in tokens if not t.startswith("DEM_")]
    n_keep = int(len(other_tokens) * partial)
    return dem_tokens + other_tokens[:n_keep]


def trajectories_to_text(trajectories: list[list[str]]) -> list[str]:
    """Превращает списки токенов в строки через пробел для TfidfVectorizer."""
    return [" ".join(traj) for traj in trajectories]



def fit_tfidf(train_texts: list[str]) -> TfidfVectorizer:
    """Обучает TF-IDF на train, токенизация по пробелу (не lowercase)."""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,  # отключаем regex-токенизацию
    )
    vectorizer.fit(train_texts)
    return vectorizer


def compute_class_centroid(
    matrix: np.ndarray,
    labels: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """Среднее TF-IDF по всем траекториям заданного класса (l2-нормировано)."""
    mask = labels == target_class
    centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


def score_trajectories(
    test_matrix: np.ndarray,
    pos_centroid: np.ndarray,
    neg_centroid: np.ndarray,
) -> np.ndarray:
    """Score = cos_sim(traj, pos) - cos_sim(traj, neg). Выше score → выше P(y=1)."""
    sim_pos = cosine_similarity(test_matrix, pos_centroid.reshape(1, -1)).ravel()
    sim_neg = cosine_similarity(test_matrix, neg_centroid.reshape(1, -1)).ravel()
    return sim_pos - sim_neg


def evaluate_task(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 100,
) -> dict:
    """Считает AUROC и AUPRC с bootstrap CI через авторские функции."""
    common_kwargs = dict(
        n_bootstraps=n_bootstraps,
        use_bag_of_little_bootstraps=False,
    )
    auroc_mean, auroc_std, _ = bootstrap_auroc_ci(
        y_true, y_score, **common_kwargs,
    )
    auprc_mean, auprc_std, _ = bootstrap_auprc_ci(
        y_true, y_score, **common_kwargs,
    )
    return {
        "auroc_mean": auroc_mean,
        "auroc_std":  auroc_std,
        "auprc_mean": auprc_mean,
        "auprc_std":  auprc_std,
        "n_pos":      int(y_true.sum()),
        "n_neg":      int(len(y_true) - y_true.sum()),
    }


def prepare_split(path: str) -> Tuple[list[list[str]], dict[str, np.ndarray]]:
    """Загружает сплит и возвращает чистые траектории + словарь {task: y_true}."""
    raw = load_split(path)
    clean_trajectories = []
    labels_per_task = {task: [] for task in OUTCOME_CLASSES}

    for traj in raw:
        labels, tokens = extract_labels_and_tokens(traj)
        if not all(task in labels for task in OUTCOME_CLASSES):
            continue
        clean_trajectories.append(tokens)
        for task in OUTCOME_CLASSES:
            labels_per_task[task].append(labels[task])

    labels_per_task = {k: np.asarray(v) for k, v in labels_per_task.items()}
    return clean_trajectories, labels_per_task


def run_baseline_for_p(
    train_trajectories: list[list[str]],
    train_labels: dict[str, np.ndarray],
    test_trajectories: list[list[str]],
    test_labels:  dict[str, np.ndarray],
    partial: float,
) -> dict[str, dict]:
    """Прогоняет полный пайплайн для одной доли траектории partial ∈ [0, 1]."""
    train_truncated = [truncate_trajectory(t, partial) for t in train_trajectories]
    test_truncated  = [truncate_trajectory(t, partial) for t in test_trajectories]

    train_texts = trajectories_to_text(train_truncated) or [""]
    test_texts  = trajectories_to_text(test_truncated)  or [""]

    vectorizer = fit_tfidf(train_texts)
    train_matrix = vectorizer.transform(train_texts)
    test_matrix  = vectorizer.transform(test_texts)

    results = {}
    for task, (neg_label_tok, pos_label_tok) in OUTCOME_CLASSES.items():
        y_train = train_labels[task]
        y_test  = test_labels[task]

        if len(np.unique(y_test)) < 2:
            results[task] = {"skipped": "test has only one class"}
            continue

        pos_centroid = compute_class_centroid(train_matrix, y_train, target_class=1)
        neg_centroid = compute_class_centroid(train_matrix, y_train, target_class=0)
        scores = score_trajectories(test_matrix, pos_centroid, neg_centroid)
        results[task] = evaluate_task(y_test, scores)

    return results


def print_results_table(results_by_p: dict[float, dict[str, dict]]) -> None:
    """Печатает компактную таблицу результатов для всех (P, задача)."""
    header = f"{'P':>5} | {'task':<15} | {'AUROC':>14} | {'AUPRC':>14} | {'n+/n-':>10}"
    print(header)
    print("-" * len(header))
    for p in sorted(results_by_p):
        for task, res in results_by_p[p].items():
            if "skipped" in res:
                print(f"{p:>5.1f} | {task:<15} | {res['skipped']}")
                continue
            auroc = f"{res['auroc_mean']:.3f} ± {res['auroc_std']:.3f}"
            auprc = f"{res['auprc_mean']:.3f} ± {res['auprc_std']:.3f}"
            counts = f"{res['n_pos']}/{res['n_neg']}"
            print(f"{p:>5.1f} | {task:<15} | {auroc:>14} | {auprc:>14} | {counts:>10}")


def main(data_dir: str) -> None:
    """Загружает train/test, гоняет baseline по всем P, печатает таблицу."""
    train_path = os.path.join(data_dir, "train.json")
    test_path  = os.path.join(data_dir, "test.json")

    print(f"Loading train from {train_path}")
    train_trajectories, train_labels = prepare_split(train_path)
    print(f"  → {len(train_trajectories)} admissions")

    print(f"Loading test from {test_path}")
    test_trajectories, test_labels = prepare_split(test_path)
    print(f"  → {len(test_trajectories)} admissions")

    print("\nClass balance in test:")
    for task in OUTCOME_CLASSES:
        y = test_labels[task]
        print(f"  {task:<15}: pos={int(y.sum())}, neg={int(len(y) - y.sum())}")

    print("\nRunning TF-IDF baseline for partial info levels:", PARTIAL_INFO_LEVELS)
    results_by_p = {}
    for partial in PARTIAL_INFO_LEVELS:
        print(f"\n--- P = {partial} ---")
        results_by_p[partial] = run_baseline_for_p(
            train_trajectories, train_labels,
            test_trajectories,  test_labels,
            partial=partial,
        )

    print("\n\n=========== RESULTS ===========\n")
    print_results_table(results_by_p)


if __name__ == "__main__":
    default_dir = "data\datasets\mimic-iv-2.2\datasets_full"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    main(data_dir)
