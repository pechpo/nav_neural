from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DIR_RE = re.compile(r"^dir_(\d+)$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dir_index(dir_key: str) -> int | None:
    match = DIR_RE.match(dir_key)
    if not match:
        return None
    return int(match.group(1))


def _mean(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return float(sum(values_list) / len(values_list))


def _global_layer(feature_meta: dict[str, Any] | None) -> int | None:
    if not feature_meta:
        return None

    part = str(feature_meta.get("part", "")).lower()
    layer = feature_meta.get("layer", None)

    if part in {"visual_embed", "visual_fc", "visual"}:
        return 0
    if part in {"rnn_hidden", "hidden"}:
        if layer is None:
            return None
        return int(layer)
    if part in {"rnn_output", "output"}:
        return 5

    if layer is None:
        return None
    return int(layer)


@dataclass(frozen=True)
class MergedDirectionSummary:
    dir_key: str
    top3_sources: str
    top3_abs_corr_mean: float | None
    top3_layer_mean: float | None
    hidden_mae: float | None
    hidden_n_features: int | None


def _format_float(value: float | None, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _sort_dir_keys(keys: Iterable[str]) -> list[str]:
    return sorted(
        set(keys),
        key=lambda k: (_dir_index(k) is None, _dir_index(k) or 0, k),
    )


def _write_csv(path: Path, summaries: list[MergedDirectionSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dir",
                "top3_sources",
                "top3_abs_corr_mean",
                "top3_layer_mean",
                "mae",
                "n_features",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.dir_key,
                    s.top3_sources,
                    "" if s.top3_abs_corr_mean is None else s.top3_abs_corr_mean,
                    "" if s.top3_layer_mean is None else s.top3_layer_mean,
                    "" if s.hidden_mae is None else s.hidden_mae,
                    "" if s.hidden_n_features is None else s.hidden_n_features,
                ]
            )


def _resolve_under(root: Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


@dataclass(frozen=True)
class NeuronCandidate:
    source: str
    abs_corr: float
    global_layer: int | None


def _collect_candidates(
    screened_by_source: dict[str, dict[str, Any]],
    dir_key: str,
) -> list[NeuronCandidate]:
    candidates: list[NeuronCandidate] = []
    for source, screened in screened_by_source.items():
        entry = screened.get(dir_key, {})
        abs_corr: list[float] = list(entry.get("abs_corr", []))
        feature_meta: list[dict[str, Any]] = list(entry.get("feature_meta", []))
        k = min(len(abs_corr), len(feature_meta))
        for i in range(k):
            corr = abs_corr[i]
            if corr is None:
                continue
            layer = _global_layer(feature_meta[i])
            candidates.append(NeuronCandidate(source=source, abs_corr=float(corr), global_layer=layer))
    return candidates


def _summarize_merged(
    *,
    screened_by_source: dict[str, dict[str, Any]],
    hidden_lasso: dict[str, Any],
    topn: int,
    prefer_sources: list[str],
) -> list[MergedDirectionSummary]:
    dir_keys = _sort_dir_keys(hidden_lasso.keys())

    summaries: list[MergedDirectionSummary] = []
    for dir_key in dir_keys:
        priority = {name: (len(prefer_sources) - i) for i, name in enumerate(prefer_sources)}
        candidates = _collect_candidates(screened_by_source, dir_key)
        candidates.sort(key=lambda c: (-c.abs_corr, -priority.get(c.source, 0), c.source))
        top_candidates = candidates[:topn]

        top3_sources = ",".join([c.source for c in top_candidates])
        top3_abs_corr_mean = _mean([c.abs_corr for c in top_candidates])
        top3_layer_mean = _mean([float(c.global_layer) for c in top_candidates if c.global_layer is not None])

        hidden_entry = hidden_lasso.get(dir_key, {})
        hidden_mae = hidden_entry.get("mae", None)
        hidden_mae = float(hidden_mae) if hidden_mae is not None else None
        hidden_n_features = hidden_entry.get("n_features", None)
        hidden_n_features = int(hidden_n_features) if hidden_n_features is not None else None

        summaries.append(
            MergedDirectionSummary(
                dir_key=dir_key,
                top3_sources=top3_sources,
                top3_abs_corr_mean=top3_abs_corr_mean,
                top3_layer_mean=top3_layer_mean,
                hidden_mae=hidden_mae,
                hidden_n_features=hidden_n_features,
            )
        )

    summaries.sort(
        key=lambda s: (
            s.hidden_mae is None,
            float("inf") if s.hidden_mae is None else s.hidden_mae,
            _dir_index(s.dir_key) or 0,
            s.dir_key,
        )
    )
    return summaries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create one merged per-direction table by selecting top-N neurons across visual/hidden/output "
            "(by Spearman abs(corr)); MAE/n_features are taken from hidden."
        )
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        default="",
        help="Directory containing analysis_out_* folders (default: directory of this script).",
    )
    parser.add_argument("--visual-dir", type=str, default="analysis_out_visual", help="Visual analysis folder.")
    parser.add_argument("--hidden-dir", type=str, default="analysis_out_hidden", help="Hidden analysis folder.")
    parser.add_argument("--output-dir", type=str, default="analysis_out_rnnoutput", help="RNN output analysis folder.")
    parser.add_argument("--topn", type=int, default=3, help="How many top neurons to average (default: 3).")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional: write the merged summary to a CSV file.",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="hidden,visual,output",
        help="Tie-break order when abs(corr) values are equal (comma-separated).",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    analysis_root = script_dir
    if args.analysis_root:
        analysis_root = _resolve_under(Path.cwd(), args.analysis_root)

    visual_dir = _resolve_under(analysis_root, args.visual_dir)
    hidden_dir = _resolve_under(analysis_root, args.hidden_dir)
    output_dir = _resolve_under(analysis_root, args.output_dir)

    hidden_lasso_path = hidden_dir / "lasso_metrics.json"
    hidden_screened_path = hidden_dir / "screened_neurons.json"
    visual_screened_path = visual_dir / "screened_neurons.json"
    output_screened_path = output_dir / "screened_neurons.json"

    if not hidden_lasso_path.exists():
        print(f"Missing {hidden_lasso_path}", file=sys.stderr)
        return 2
    if not hidden_screened_path.exists():
        print(f"Missing {hidden_screened_path}", file=sys.stderr)
        return 2

    hidden_lasso = _load_json(hidden_lasso_path)

    screened_by_source: dict[str, dict[str, Any]] = {"hidden": _load_json(hidden_screened_path)}
    if visual_screened_path.exists():
        screened_by_source["visual"] = _load_json(visual_screened_path)
    if output_screened_path.exists():
        screened_by_source["output"] = _load_json(output_screened_path)

    prefer_sources = [s.strip() for s in str(args.prefer).split(",") if s.strip()]
    if not prefer_sources:
        prefer_sources = ["hidden", "visual", "output"]

    summaries = _summarize_merged(
        screened_by_source=screened_by_source,
        hidden_lasso=hidden_lasso,
        topn=args.topn,
        prefer_sources=prefer_sources,
    )

    rows = []
    for s in summaries:
        rows.append(
            [
                s.dir_key,
                s.top3_sources,
                _format_float(s.top3_abs_corr_mean, digits=4),
                _format_float(s.top3_layer_mean, digits=2),
                _format_float(s.hidden_mae, digits=4),
                "" if s.hidden_n_features is None else str(s.hidden_n_features),
            ]
        )

    print(
        _markdown_table(
            ["dir", "top3_sources", "top3_abs_corr_mean", "top3_layer_mean", "mae", "n_features"],
            rows,
        )
    )

    csv_out = Path(args.csv_out).expanduser() if args.csv_out else None
    if csv_out is not None:
        out_path = csv_out
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / "merged_summary.csv"
        _write_csv(out_path, summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
