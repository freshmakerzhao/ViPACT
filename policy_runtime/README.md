# policy_runtime

Low-level policy execution runtime (the "cerebellum" side).

Current status:
- `ACTPolicyRuntime` provides a minimal inference wrapper around ACT.
- Input type: `PolicyStepInput` from `vipact_interfaces.py`
- Output type: `PolicyStepOutput` from `vipact_interfaces.py`

Current mask rule (aligned with project mainline):
- Only `cockpit` camera can use non-zero mask channel.
- Other cameras use zero-filled mask channel.

