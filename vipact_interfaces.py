from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BrainOutput:
    """
    High-level decision output used by policy runtime.
    """

    target_id: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyStepInput:
    """
    Unified per-step input for low-level ACT runtime.
    """

    qpos: Any
    images_by_camera: Dict[str, Any]
    brain_output: BrainOutput
    static_mask_by_camera: Optional[Dict[str, Any]] = None
    step_id: int = 0


@dataclass
class PolicyStepOutput:
    """
    Unified per-step output from low-level ACT runtime.
    """

    action: Any
    meta: Dict[str, Any] = field(default_factory=dict)

