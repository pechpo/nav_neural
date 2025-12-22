"""Spearman correlation utilities (SciPy-backed)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import rankdata, zscore


def spearman_corr(
    activations: np.ndarray,
    targets: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    """Compute Spearman correlation matrix (D x K).

    Args:
        activations: shape (N, D)
        targets: shape (N, K)
        chunk_size: number of neurons per chunk

    Returns:
        corr: shape (D, K)
    """
    activations = np.asarray(activations)
    targets = np.asarray(targets)
    if activations.ndim != 2 or targets.ndim != 2:
        raise ValueError("activations and targets must be 2D arrays")
    if activations.shape[0] != targets.shape[0]:
        raise ValueError("activations and targets must share N dimension")

    n, d = activations.shape
    k = targets.shape[1]
    if n < 2:
        raise ValueError("Need at least 2 samples to compute correlation")

    target_ranks = np.apply_along_axis(rankdata, 0, targets).astype(np.float64)
    target_ranks = zscore(target_ranks, axis=0, ddof=1)
    target_ranks = np.nan_to_num(target_ranks, nan=0.0)

    corr = np.zeros((d, k), dtype=np.float64)
    denom = float(n - 1)

    for start in range(0, d, chunk_size):
        end = min(start + chunk_size, d)
        chunk = activations[:, start:end]
        chunk_ranks = np.apply_along_axis(rankdata, 0, chunk).astype(np.float64)
        chunk_ranks = zscore(chunk_ranks, axis=0, ddof=1)
        chunk_ranks = np.nan_to_num(chunk_ranks, nan=0.0)

        corr[start:end, :] = (chunk_ranks.T @ target_ranks) / denom

    return corr


def screen_by_correlation(
    corr: np.ndarray,
    topk: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Screen neurons by absolute correlation for each target direction.

    Returns:
        indices: object array of length K, each entry is 1D indices
        scores: object array of length K, each entry is abs(corr) sorted
    """
    corr = np.asarray(corr)
    if corr.ndim != 2:
        raise ValueError("corr must be 2D")
    d, k = corr.shape

    indices = np.empty(k, dtype=object)
    scores = np.empty(k, dtype=object)

    for j in range(k):
        abs_corr = np.abs(corr[:, j])
        order = np.argsort(-abs_corr)
        if threshold is not None:
            mask = abs_corr[order] >= threshold
            order = order[mask]
        if topk is not None:
            order = order[:topk]
        indices[j] = order
        scores[j] = abs_corr[order]

    return indices, scores
