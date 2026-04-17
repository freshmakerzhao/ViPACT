from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from interfaces import TargetSelectionResult


@dataclass
class SelectionRules:
    text_prompt: str
    axis: str
    reverse: bool
    target_rank: int


def _extract_candidates(dino_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Support both structures:
    # - {"detections": [...]}
    # - {"candidates": [...]}
    # - {"det_debug": {"candidates": [...]}}
    if "detections" in dino_json and isinstance(dino_json["detections"], list):
        return dino_json["detections"]
    if "candidates" in dino_json and isinstance(dino_json["candidates"], list):
        return dino_json["candidates"]
    det_debug = dino_json.get("det_debug", {})
    if isinstance(det_debug, dict) and isinstance(det_debug.get("candidates"), list):
        return det_debug.get("candidates", [])
    return []


def _center_xy(box_xyxy: List[float]) -> (float, float):
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def find_target_location(
    dino_json: Dict[str, Any],
    rules: Dict[str, Any],
    *,
    tolerance: float = 20.0,
) -> Dict[str, Any]:
    """Select one target from DINO candidates by parsed spatial rules.

    Simplified strategy:
    1) score filter
    2) sort by axis (x/y) and direction (reverse)
    3) pick N-th item directly (target_rank)
    """
    axis = str(rules.get("axis", "x")).lower()
    reverse = bool(rules.get("reverse", False))
    target_rank = max(1, int(rules.get("target_rank", 1)))
    if axis not in ("x", "y"):
        axis = "x"

    candidates = _extract_candidates(dino_json)
    valid_boxes: List[Dict[str, Any]] = []

    for c in candidates:
        score = float(c.get("score", 0.0))
        box = c.get("box_xyxy", [0, 0, 0, 0])
        center_x, center_y = _center_xy(box)
        primary_val = center_x if axis == "x" else center_y
        valid_boxes.append(
            {
                "index": int(c.get("index", -1)),
                "rank": int(c.get("rank", -1)),
                "score": score,
                "box_xyxy": [float(v) for v in box],
                "center_x": float(center_x),
                "center_y": float(center_y),
                "primary_val": float(primary_val),
            }
        )

    if not valid_boxes:
        return {
            "ok": False,
            "reason": "no_valid_boxes",
            "selected": None,
            "sorted_boxes": [],
        }

    valid_boxes.sort(key=lambda x: x["primary_val"], reverse=reverse)
    target_idx = target_rank - 1
    if target_idx >= len(valid_boxes):
        return {
            "ok": False,
            "reason": f"target_rank_out_of_range(target_rank={target_rank}, num_boxes={len(valid_boxes)})",
            "selected": None,
            "sorted_boxes": valid_boxes,
        }

    selected = valid_boxes[target_idx]
    return {
        "ok": True,
        "reason": "success",
        "selected": selected,
        "sorted_boxes": valid_boxes,
        "note": "tolerance is ignored in simplified selector",
    }


def select_target_typed(
    dino_json: Dict[str, Any],
    rules: Dict[str, Any],
    *,
    tolerance: float = 20.0,
) -> TargetSelectionResult:
    raw = find_target_location(
        dino_json=dino_json,
        rules=rules,
        tolerance=float(tolerance),
    )
    return TargetSelectionResult(
        ok=bool(raw.get("ok", False)),
        reason=str(raw.get("reason", "")),
        selected=raw.get("selected", {}) or {},
        meta={
            "sorted_boxes": raw.get("sorted_boxes", []),
            "raw": raw,
        },
    )
