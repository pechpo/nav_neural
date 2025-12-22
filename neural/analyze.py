"""Spearman screening + L1 regression pipeline for neuron-direction analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from .io_utils import ensure_dir, load_npz, save_json
from .models.lasso import fit_lasso
from .models.spearman import spearman_corr, screen_by_correlation


def _parse_direction_names(raw: Optional[str], k: int) -> List[str]:
    if raw is None:
        return [f"dir_{i}" for i in range(k)]
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if len(names) != k:
        raise ValueError(f"Expected {k} direction names, got {len(names)}")
    return names


def _build_screen_payload(
    corr: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    dir_names: List[str],
    feature_key: str,
    meta: dict,
) -> dict:
    payload = {}
    for j, name in enumerate(dir_names):
        idx = indices[j]
        feature_meta = _describe_feature_indices(idx, feature_key, meta)
        payload[name] = {
            "indices": idx.tolist(),
            "abs_corr": scores[j].tolist(),
            "signed_corr": corr[idx, j].tolist() if idx.size > 0 else [],
            "feature_meta": feature_meta,
        }
    return payload


def _describe_feature_indices(indices: np.ndarray, feature_key: str, meta: dict) -> List[dict]:
    if indices is None or len(indices) == 0:
        return []
    key = feature_key
    if key in ("activations", "activations_hidden", "hidden"):
        part = "rnn_hidden"
        hidden_size = (
            meta.get("rnn_output_dim")
            or meta.get("rnn_hidden_size")
            or meta.get("recurrent_hidden_size")
        )
        out = []
        for idx in indices.tolist():
            if hidden_size:
                layer = int(idx // hidden_size) + 1
                unit = int(idx % hidden_size)
            else:
                layer = None
                unit = int(idx)
            out.append({"part": part, "layer": layer, "unit": unit})
        return out
    if key in ("activations_rnn_output", "rnn_output"):
        return [{"part": "rnn_output", "layer": 1, "unit": int(idx)} for idx in indices.tolist()]
    if key in ("activations_visual_embed", "visual_embed"):
        return [{"part": "visual_embed", "layer": 1, "unit": int(idx)} for idx in indices.tolist()]
    return [{"part": "unknown", "layer": None, "unit": int(idx)} for idx in indices.tolist()]


def _fit_lasso_per_direction(
    activations: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    dir_names: List[str],
    alpha: float,
    max_iter: int,
    tol: float,
) -> dict:
    n, d = activations.shape
    k = targets.shape[1]
    weights = np.zeros((k, d), dtype=np.float64)
    intercepts = np.zeros(k, dtype=np.float64)
    metrics = {}

    for j, name in enumerate(dir_names):
        idx = indices[j]
        y = targets[:, j]
        if idx.size == 0:
            intercepts[j] = float(y.mean())
            metrics[name] = {
                "n_features": 0,
                "r2": 0.0,
                "mae": float(np.mean(np.abs(y - y.mean()))),
            }
            continue
        x = activations[:, idx]
        result = fit_lasso(x, y, alpha=alpha, max_iter=max_iter, tol=tol)
        weights[j, idx] = result.weights
        intercepts[j] = result.intercept
        metrics[name] = {
            "n_features": int(idx.size),
            "n_iter": int(result.n_iter),
            "r2": float(result.r2),
            "mae": float(result.mae),
        }

    return {"weights": weights, "intercepts": intercepts, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Spearman screening + L1 regression")
    parser.add_argument("--data", default="neural/result.npz", help="Path to .npz with activations and distances")
    parser.add_argument("--out", default="neural/analysis_out", help="Output directory")
    parser.add_argument("--feature-key", default="activations_hidden", help="Key for activations in npz")
    parser.add_argument("--target-key", default="distances", help="Key for target distances in npz")
    parser.add_argument("--direction-names", default=None, help="Comma-separated direction names")
    parser.add_argument("--topk", type=int, default=50, help="Top-k neurons per direction (<=0 to disable)")
    parser.add_argument("--threshold", type=float, default=None, help="Min abs Spearman to keep")
    parser.add_argument("--chunk-size", type=int, default=256, help="Spearman chunk size")
    parser.add_argument("--lasso-alpha", type=float, default=0.01, help="L1 regularization strength")
    parser.add_argument("--lasso-max-iter", type=int, default=2000, help="Max iterations")
    parser.add_argument("--lasso-tol", type=float, default=1e-4, help="Convergence tolerance")

    args = parser.parse_args()

    activations, targets, meta = load_npz(args.data, args.feature_key, args.target_key)

    if activations.ndim != 2 or targets.ndim != 2:
        raise ValueError("activations and targets must be 2D arrays")
    if activations.shape[0] != targets.shape[0]:
        raise ValueError("activations and targets must share N dimension")

    n, d = activations.shape
    k = targets.shape[1]
    dir_names = _parse_direction_names(args.direction_names, k)

    out_dir = ensure_dir(args.out)

    corr = spearman_corr(activations, targets, chunk_size=args.chunk_size)
    np.save(out_dir / "spearman_corr.npy", corr)

    topk = args.topk if args.topk and args.topk > 0 else None
    indices, scores = screen_by_correlation(corr, topk=topk, threshold=args.threshold)
    screen_payload = _build_screen_payload(corr, indices, scores, dir_names, args.feature_key, meta)
    save_json(out_dir / "screened_neurons.json", screen_payload)

    lasso = _fit_lasso_per_direction(
        activations,
        targets,
        indices,
        dir_names,
        alpha=args.lasso_alpha,
        max_iter=args.lasso_max_iter,
        tol=args.lasso_tol,
    )
    np.save(out_dir / "lasso_weights.npy", lasso["weights"])
    np.save(out_dir / "lasso_intercepts.npy", lasso["intercepts"])
    save_json(out_dir / "lasso_metrics.json", lasso["metrics"])

    config = {
        "data": str(Path(args.data).resolve()),
        "feature_key": args.feature_key,
        "target_key": args.target_key,
        "n_samples": int(n),
        "n_neurons": int(d),
        "n_directions": int(k),
        "direction_names": dir_names,
        "spearman_chunk_size": int(args.chunk_size),
        "screen_topk": int(topk) if topk is not None else None,
        "screen_threshold": float(args.threshold) if args.threshold is not None else None,
        "lasso_alpha": float(args.lasso_alpha),
        "lasso_max_iter": int(args.lasso_max_iter),
        "lasso_tol": float(args.lasso_tol),
        "npz_keys": meta.get("keys", []),
        "hidden_dim": meta.get("hidden_dim"),
        "rnn_output_dim": meta.get("rnn_output_dim"),
        "visual_embed_dim": meta.get("visual_embed_dim"),
        "rnn_num_layers": meta.get("rnn_num_layers"),
        "visible_dir_mask": meta.get("visible_dir_mask"),
        "visible_dir_indices": meta.get("visible_dir_indices"),
        "invisible_dir_indices": meta.get("invisible_dir_indices"),
    }
    save_json(out_dir / "config.json", config)


if __name__ == "__main__":
    main()
