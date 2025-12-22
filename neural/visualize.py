"""Heatmap visualizations for neuron-direction relationships."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .io_utils import ensure_dir, load_npz


def _load_analysis_meta(analysis_out: Path) -> Dict[str, object]:
    meta_path = analysis_out / "config.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_screened(analysis_out: Path) -> Dict[str, dict]:
    path = analysis_out / "screened_neurons.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing screened_neurons.json in {analysis_out}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_dir_index(name: str) -> Optional[int]:
    try:
        if name.startswith("dir_"):
            return int(name.split("_", 1)[1])
    except Exception:
        return None
    return None


def _bin_distances(distances: np.ndarray, n_bins: int, max_range: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, float(max_range), int(n_bins) + 1, dtype=np.float32)
    idx = np.digitize(distances, edges, right=False) - 1
    idx = np.clip(idx, 0, int(n_bins) - 1)
    return idx, edges


def _compute_heatmap(
    activations: np.ndarray,
    dist_bin_idx: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_dirs = dist_bin_idx.shape
    heat = np.full((n_dirs, n_bins), np.nan, dtype=np.float32)
    counts = np.zeros((n_dirs, n_bins), dtype=np.int32)
    for j in range(n_dirs):
        bins = dist_bin_idx[:, j]
        for b in range(n_bins):
            mask = bins == b
            if not np.any(mask):
                continue
            heat[j, b] = float(np.mean(activations[mask]))
            counts[j, b] = int(mask.sum())
    return heat, counts


def _interp_directions(heat: np.ndarray, out_dirs: int) -> np.ndarray:
    n_dirs, n_bins = heat.shape
    if out_dirs <= n_dirs:
        return heat
    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    full = np.linspace(0.0, 2.0 * np.pi, out_dirs, endpoint=False)
    interp = np.full((out_dirs, n_bins), np.nan, dtype=np.float32)
    for b in range(n_bins):
        vals = heat[:, b]
        valid = ~np.isnan(vals)
        if np.sum(valid) == 0:
            continue
        ang = angles[valid]
        v = vals[valid]
        order = np.argsort(ang)
        ang = ang[order]
        v = v[order]
        ang_ext = np.concatenate([ang, ang + 2.0 * np.pi])
        v_ext = np.concatenate([v, v])
        interp[:, b] = np.interp(full, ang_ext, v_ext)
    return interp


def _describe_feature(
    entry: dict,
    idx_pos: int,
    idx: int,
    feature_key: str,
    meta: Dict[str, object],
) -> str:
    fm_list = entry.get("feature_meta")
    if isinstance(fm_list, list) and idx_pos < len(fm_list):
        fm = fm_list[idx_pos]
        part = fm.get("part")
        layer = fm.get("layer")
        unit = fm.get("unit", idx)
        if layer is not None:
            return f"{part}-L{layer}-U{unit}"
        return f"{part}-U{unit}"
    if feature_key in ("activations_hidden", "hidden"):
        hidden = meta.get("rnn_output_dim")
        if hidden:
            layer = int(idx // int(hidden)) + 1
            unit = int(idx % int(hidden))
            return f"rnn_hidden-L{layer}-U{unit}"
        return f"rnn_hidden-U{idx}"
    if feature_key in ("activations_rnn_output", "rnn_output"):
        return f"rnn_output-U{idx}"
    if feature_key in ("activations_visual_embed", "visual_embed"):
        return f"visual_embed-U{idx}"
    return f"unit-{idx}"


def _plot_heatmap(
    heat: np.ndarray,
    edges: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    n_dirs, n_bins = heat.shape
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_dirs + 1, endpoint=True)
    r_edges = edges
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="black")
    pcm = ax.pcolormesh(theta_grid, r_grid, heat.T, cmap=cmap, shading="auto")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, va="bottom")
    ax.set_ylabel("distance (m)")
    plt.colorbar(pcm, ax=ax, pad=0.1, label="mean activation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize neuron-direction heatmaps")
    parser.add_argument("--analysis-out", default="neural/analysis_out", help="Analysis output directory")
    parser.add_argument("--data", default=None, help="Override data .npz path")
    parser.add_argument("--feature-key", default=None, help="Override feature key")
    parser.add_argument("--topk", type=int, default=3, help="Top-k neurons per direction")
    parser.add_argument("--bins", type=int, default=30, help="Distance bins")
    parser.add_argument("--interp-dirs", type=int, default=360, help="Interpolate directions to this count")
    args = parser.parse_args()

    analysis_out = Path(args.analysis_out)
    ensure_dir(analysis_out)

    meta = _load_analysis_meta(analysis_out)
    data_path = Path(args.data or meta.get("data", "neural/result.npz"))
    feature_key = str(args.feature_key or meta.get("feature_key", "activations_hidden"))

    activations, distances, meta_npz = load_npz(data_path, feature_key, "distances")
    num_dirs = int(distances.shape[1])
    max_range = float(meta_npz.get("max_range") or float(np.nanmax(distances)))

    screened = _load_screened(analysis_out)
    dir_names = [f"dir_{i}" for i in range(num_dirs)]

    dist_bins, edges = _bin_distances(distances, args.bins, max_range)
    out_dir = analysis_out

    for name in dir_names:
        entry = screened.get(name)
        if not isinstance(entry, dict):
            continue
        indices = entry.get("indices", [])
        if not isinstance(indices, list) or not indices:
            continue
        top_indices = indices[: int(args.topk)]
        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            acts = activations[:, idx]
            heat, _ = _compute_heatmap(acts, dist_bins, args.bins)
            heat_full = _interp_directions(heat, int(args.interp_dirs))
            feature_label = _describe_feature(entry, rank - 1, idx, feature_key, meta_npz)
            title = f"{name} top{rank}: {feature_label}"
            out_name = f"heat_{name}_rank{rank}_{feature_label}.png"
            _plot_heatmap(heat_full, edges, title, out_dir / out_name)


if __name__ == "__main__":
    main()
