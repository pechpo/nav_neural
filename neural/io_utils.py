"""Load and save utilities for the analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np


def load_npz(path: Union[str, Path], feature_key: str, target_key: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    p = Path(path)
    data = np.load(p, allow_pickle=False)
    if feature_key not in data or target_key not in data:
        keys = ", ".join(sorted(data.files))
        raise KeyError(f"Missing keys in npz. Found: {keys}")
    activations = data[feature_key]
    targets = data[target_key]
    meta = {"keys": list(data.files)}
    meta_path = p.with_suffix(".json")
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta.update(json.load(f))
        except Exception:
            pass
    return activations, targets, meta


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
