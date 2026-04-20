from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _to_serializable(obj: Any):
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class ArtifactLogger:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, tag: str) -> Path:
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = self.root / f"{stamp}_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_json(self, path: Path, obj: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(obj), f, ensure_ascii=False, indent=2)
        return path

    def save_mask(self, path: Path, mask_u8: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(mask_u8)
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape={arr.shape}")
        out = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(out, mode="L").save(path)
        return path

