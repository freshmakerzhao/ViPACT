from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PipelineRunSummary:
    ok: bool
    mode: str
    episode_id: int
    run_dir: str
    strict_success: Any = None
    episode_return: Any = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

