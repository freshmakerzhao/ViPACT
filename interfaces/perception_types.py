from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DetectionCandidate:
    """
    Single object candidate from detector.
    """

    index: int
    score: float
    label: str
    box_xyxy: List[float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """
    Detector output containing all candidates.
    """

    ok: bool
    candidates: List[DetectionCandidate] = field(default_factory=list)
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskResult:
    """
    Binary mask generation output.
    """

    ok: bool
    mask: Optional[np.ndarray] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

