"""L1-regularized linear regression (sklearn-backed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class LassoResult:
    weights: np.ndarray
    intercept: float
    n_iter: int
    r2: float
    mae: float


def fit_lasso(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> LassoResult:
    """Fit Lasso with sklearn and return weights in original feature scale."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2D and y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must share N dimension")

    scaler = StandardScaler(with_mean=True, with_std=True)
    xs = scaler.fit_transform(x)

    model = Lasso(alpha=float(alpha), max_iter=int(max_iter), tol=float(tol), fit_intercept=True)
    model.fit(xs, y)

    coef_scaled = model.coef_.astype(np.float64, copy=False)
    scale = scaler.scale_.astype(np.float64, copy=False)
    mean = scaler.mean_.astype(np.float64, copy=False)

    weights = coef_scaled / scale
    intercept = float(model.intercept_ - np.dot(coef_scaled, mean / scale))

    y_hat = x @ weights + intercept
    y_mean = float(np.mean(y))
    denom = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - float(np.sum((y - y_hat) ** 2)) / denom if denom > 0 else 0.0
    mae = float(np.mean(np.abs(y - y_hat)))

    return LassoResult(
        weights=weights,
        intercept=intercept,
        n_iter=int(getattr(model, "n_iter_", max_iter)),
        r2=r2,
        mae=mae,
    )
