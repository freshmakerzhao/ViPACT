from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class InstructionParseResult:
    """
    Structured instruction parsed by high-level brain.
    """

    text_prompt: str
    axis: str = "x"
    reverse: bool = False
    target_rank: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSelectionResult:
    """
    Selected detection candidate after ranking/selection logic.
    """

    ok: bool
    reason: str = ""
    selected: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

