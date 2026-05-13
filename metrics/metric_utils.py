import io
import gc
import re
import subprocess

import numpy as cp
import numpy as np

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from typing import Callable
from tqdm import tqdm
from PIL import Image

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
)


def log_figure_to_board(
    fig: plt.Figure,
    title: str,
    logger: pl.loggers.Logger,
    global_step: int=0,
) -> None:
    """ Log a matplotlib figure to tensorboard, as an image
    """
    print("Logging figure %s to the board" % title)
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, dpi=300, format="png")
    buffer.seek(0)
    with Image.open(buffer) as img: image = np.array(img)


def clean_memory_fn():
    """Clean CPU memory after model run."""
    gc.collect()


def clean_memory():
    """ Decorator to clean memory before and after a function call
    """
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            clean_memory_fn()
            result = original_function(*args, **kwargs)
            clean_memory_fn()
            return result
        return wrapper
    return decorator


@clean_memory()
def compute_reduced_representation(
    data: np.ndarray,
    dim: int=2,
    tsne_metric: str="cosine",
    rdm_metric: str=None,
) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
        Similarity metric can be None, "correlation", or "euclidean"
    """
    print("Reducing %s samples from dim %s to dim %s" % (data.shape[:2] + (dim,)))
    
    # If required, represent data based on sample similarity instead of data
    if rdm_metric is not None:
        data = rdm_with_dask(data, rdm_metric, retrieve_dim=True)
    
    # Return dimensionality-reduced data using t-SNE
    params = {
        "n_components": dim,
        "n_iter": 100_000,
        "n_iter_without_progress": 1_000,
        "method": "barnes_hut",
        "metric": tsne_metric,
        "perplexity": 50.0,  # Canny-lab's t-SNE default
        "n_neighbors": 32,  # Canny-lab's t-SNE default
        "learning_rate_method": "none",  # Canny-lab's t-SNE default
    }
    tsne = TSNE(**params)
    return tsne.fit_transform(data)


@clean_memory()
def rdm(
    data: np.ndarray,
    metric: str="correlation",
    retrieve_dim: bool=True,
) -> np.ndarray:
    """ Compute a new data representation based on sample-pair similiarities
        *** Not used in metric_utils.py for now! ***
    """
    print("Computing representational sample dissimilarity matrix")
    distance_matrix = cdist(data, metric).get()
    
    if retrieve_dim:
        print("Running PCA to retrieve original dimension from RDM")
        pca = PCA(n_components=data.shape[1])
        return pca.fit_transform(distance_matrix)
    else:
        return distance_matrix
    
    
@clean_memory()
def rdm_with_dask(
    data: np.ndarray,
    metric: str="correlation",  # "mahalanobis"
    retrieve_dim: bool=True,
    n_rows_per_gpu_chunk: int=10_000,
) -> np.ndarray:
    """ Compute a new data representation based on sample-pair similiarities
    """
    # Figure out how many GPUs can be used for RDM computation
    chunk_size = n_rows_per_gpu_chunk * max(n_rows_per_gpu_chunk, data.shape[1])
    used_memory = chunk_size * data.itemsize / (10 ** 9) * 4
    usable_devices = [k for k, v in get_gpu_memory().items() if v > used_memory]
    
    
def bootstrap(
    trues: cp.ndarray,
    scores: cp.ndarray,
    metric_fn: Callable,
    n_bootstraps: int=100,
    **kwargs,
) -> list[float]:
    """ Classic bootstrapping procedure (using cupy arrays)
    """
    n = len(trues)
    bootstrapped_metrics = []
    for _ in tqdm(range(n_bootstraps), desc=" ----- Bootstrapping", leave=False):
        
        # Compute metric based on subsample
        sampled_trues, sampled_scores = bootstrap_sampling(trues, scores)
        metric = metric_fn(sampled_trues, sampled_scores, **kwargs)
            
        # Depending on metric_fn, metric can be float or cupy.ndarray
        if isinstance(metric, cp.ndarray): metric = metric.item()
        bootstrapped_metrics.append(metric)
    
    return bootstrapped_metrics


def bag_of_little_bootstraps(
    trues: cp.ndarray,
    scores: cp.ndarray,
    metric_fn: Callable,
    n_bootstraps: int=100,
    n_subsamples: int=100,
    subsample_size: int=10_000,
    **kwargs,
) -> list[float]:
    """ More memory efficient bag of little bootstraps procedure
    """
    bootstrapped_metrics = []
    for _ in tqdm(range(n_subsamples), desc=" ----- BLB algorithm", leave=False):
        
        # Sample a subset of the dataset
        subset_trues, subset_scores = bootstrap_sampling(
            trues, scores, subsample_size=subsample_size,
        )
        
        # Apply standard bootstrap on this subset
        subset_metrics = bootstrap(
            subset_trues, subset_scores, metric_fn, n_bootstraps, **kwargs,
        )
        
        # Record results
        bootstrapped_metrics.extend(subset_metrics)
        
    return bootstrapped_metrics


def bootstrap_sampling(
    trues: cp.ndarray,
    scores: cp.ndarray,
    subsample_size: int=None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """ Sampling function for bootstrapping and BLB algorithms
    """
    n_samples = len(trues)
    while True:
        
        # Random sampling with or without replacement
        if subsample_size is None:
            indices = cp.random.randint(0, n_samples, n_samples)
        else:
            indices = cp.random.choice(n_samples, subsample_size, replace=False)
        
        # Continue only if at least one negative and one positive class
        sampled_trues = trues[indices]
        if len(cp.unique(sampled_trues)) >= 2: break
    
    # Subsample scores array with corresponding indices
    sampled_scores = scores[indices]
    
    return sampled_trues, sampled_scores


def bootstrap_metric_ci(
    trues: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable,
    n_bootstraps: int=100,
    use_bag_of_little_bootstraps: bool=True,
    blb_params: dict={"n_subsamples": 100, "subsample_size": 10_000},
    **kwargs,
) -> tuple[float]:
    """ Compute statistics of a given metric using the given bootstrap algorithm
    """
    # Compute bootstrapped scores using the required algorithm
    trues_, scores_ = cp.array(trues), cp.array(scores)  # computations on GPU
    if not use_bag_of_little_bootstraps:
        metrics = bootstrap(trues_, scores_, metric_fn, n_bootstraps, **kwargs)
    else:
        metrics = bag_of_little_bootstraps(
            trues_, scores_, metric_fn, n_bootstraps, **blb_params, **kwargs,
        )
    
    # Compute boostrapped statistics
    mean = np.mean(metrics)
    std_dev = np.std(metrics)
    std_error_mean = std_dev / np.sqrt(len(metrics))
    
    return mean, std_dev, std_error_mean


def bootstrap_auroc_ci(
        trues: np.ndarray,
        scores: np.ndarray,
        **kwargs,
) -> tuple[float]:
    """Bootstrapping procedure for AUROC computation."""

    return bootstrap_metric_ci(
        trues,
        scores,
        roc_auc_score,
        **kwargs,
    )
    

def bootstrap_auprc_ci(
    trues: np.ndarray,
    scores: np.ndarray,
    **kwargs,
) -> tuple[float]:
    """Bootstrapping procedure for AUPRC computation."""

    return bootstrap_metric_ci(
        trues,
        scores,
        pr_auc_score,
        **kwargs,
    )


def pr_auc_score(
    trues: np.ndarray,
    scores: np.ndarray,
    **kwargs,
) -> float:
    """Compute AUPRC on CPU."""

    precision, recall, _ = precision_recall_curve(
        trues,
        scores,
        **kwargs,
    )

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def bootstrap_rate_reduction_ci(
    labels: list,
    embeddings: np.ndarray,
    **kwargs,
) -> tuple[float]:
    """Bootstrapping procedure for rate reduction computation."""

    classes = sorted(list(set(labels)))
    label_indices = np.array([classes.index(l) for l in labels])

    return bootstrap_metric_ci(
        label_indices,
        embeddings,
        rate_reduction_score,
        use_bag_of_little_bootstraps=False,
        **kwargs,
    )


def rate_reduction_score(label_indices, embeddings):
    """ Function that computes rate reduction for a set of embeddings
    """
    partitions = compute_partitions(label_indices)
    return rate_reduction(embeddings, partitions)
    
    
def compute_partitions(
    label_indices: cp.ndarray,  # (n_samples,)
)-> cp.ndarray:  # (n_classes, n_samples, n_samples)
    """ Generate class membership matrix from a vector of labels
    """
    n_classes, n_samples = len(cp.unique(label_indices)), len(label_indices)
    partitions = cp.zeros((n_classes, n_samples, n_samples), dtype=cp.float32)
    for j, k in enumerate(label_indices):
        partitions[k, j, j] = 1.0
    return partitions


def covariance(samples: cp.ndarray) -> np.ndarray:
    """ Compute covariace matrix given sample features
    """
    return samples.T @ samples
    

def logdet(matrix: cp.ndarray) -> float:
    """ Compute logarithm of determinant of a matrix
    """
    sign, log_det = cp.linalg.slogdet(matrix)
    return sign * log_det


def rate_distortion(
    data: cp.ndarray,  # (n_samples, n_features)
    epsilon: float  # precision parameter
) -> float:  # rate distortion
    """ Compute non-asymptotic rate distortion for finite samples
    """
    n_samples, n_features = data.shape
    identity_matrix = cp.eye(n_features)
    scaling_factor = n_features / (n_samples * epsilon)
    rate_matrix = identity_matrix + scaling_factor * covariance(data)
    return 1 / 2 * logdet(rate_matrix)


def rate_reduction(
    samples: cp.ndarray,  # (n_samples, n_features)
    partitions: cp.ndarray,  # (n_classes, n_samples, n_samples)
    epsilon: float=0.01,  # precision parameter
    normalize: bool=False,  # normalize data or not
) -> float:  # rate reduction
    """ Compute rate reduction as the reduction between rate distortion as
        computed on the whole data and computed for each class separately
    """
    # Normalize data if required
    if normalize:
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        
    # Compute total rate distortion
    n_samples, _ = samples.shape
    total_r = rate_distortion(samples, epsilon)
    
    # Compute total and mean-class rate distortions
    mean_class_r = 0
    for class_partition in partitions:
        class_samples = samples[cp.diag(class_partition) == 1]
        n_class_samples = class_samples.shape[0]
        if n_class_samples == 0: continue
        class_rate = rate_distortion(class_samples, epsilon)
        mean_class_r += n_class_samples / n_samples * class_rate
    
    # Return rate reduction
    return total_r - mean_class_r


def rate_reduction_ma(
    samples: cp.ndarray,  # (n_samples, n_features)
    partitions: cp.ndarray,  # (n_classes, n_samples, n_samples)
    epsilon: float=0.01,  # precision parameter
    normalize: bool=True,  # normalize data or not
) -> float:  # rate reduction
    """ Almost copy / pasted from https://github.com/Ma-Lab-Berkeley/ReduNet
        Used to check that my own way of computing rate reduction is correct
    """
    # Normalize data if required
    if normalize:
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        
    # Compute expansion rate
    n_samples, n_features = samples.shape
    identity = np.eye(n_features)
    scaling_factor = n_features / (n_samples * epsilon)
    loss_expd = logdet(scaling_factor * covariance(samples) + identity) / 2.0
    
    # Compute compression rate
    loss_comp = 0.0
    for class_partition in partitions:
        class_samples = samples[cp.diag(class_partition) == 1]
        n_class_samples = class_samples.shape[0]
        if n_class_samples == 0: continue
        class_scaling_factor = n_features / (n_class_samples * epsilon)
        class_logdet = logdet(
            identity + class_scaling_factor * covariance(class_samples)
        )
        loss_comp += class_logdet * n_class_samples / (2 * n_samples)
    
    # Return rate reduction
    return loss_expd - loss_comp


def get_gpu_memory() -> dict[int, float]:
    """ Return GPU indices sorted by available memory, in Gb
    """
    try:
        # Querying GPU memory details
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True
        )

        # Parsing the output to extract memory information
        gpu_memory_info = {}
        for line in output.strip().split("\n"):
            match = re.match(r"(\d+), (\d+)", line)
            if match:
                index, free_memory = map(int, match.groups())
                gpu_memory_info[index] = free_memory / 1024

        # Return GPU indices with available memory (in Gb)
        return gpu_memory_info
    
    except subprocess.CalledProcessError:
        return []
