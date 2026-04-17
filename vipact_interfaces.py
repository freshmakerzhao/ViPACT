"""
Compatibility layer.

Keep old imports working:
    from vipact_interfaces import BrainOutput, PolicyStepInput, PolicyStepOutput

New code should import from:
    interfaces.policy_types
"""

from interfaces.policy_types import BrainOutput, PolicyStepInput, PolicyStepOutput

__all__ = ["BrainOutput", "PolicyStepInput", "PolicyStepOutput"]
