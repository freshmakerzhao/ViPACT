# policy_runtime

Low-level policy execution runtime (the "cerebellum" side).

Current status:
- `ACTPolicyRuntime` provides a minimal inference wrapper around ACT.
- Input type: `PolicyStepInput` from `interfaces.policy_types` (old `vipact_interfaces.py` import still compatible)
- Output type: `PolicyStepOutput` from `interfaces.policy_types` (old `vipact_interfaces.py` import still compatible)

Current mask rule (aligned with project mainline):
- Only `cockpit` camera can use non-zero mask channel.
- Other cameras use zero-filled mask channel.
