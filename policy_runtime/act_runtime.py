from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange

from policy import ACTPolicy
from interfaces import PolicyStepInput, PolicyStepOutput


def _build_act_policy(policy_config: Dict[str, Any]) -> ACTPolicy:
    # ACTPolicy internally parses argv in this repo; isolate runtime call path.
    argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0]]
    policy = ACTPolicy(policy_config)
    sys.argv = argv_backup
    return policy


@dataclass
class ACTPolicyRuntime:
    """
    Minimal ACT inference runtime for ViPACT integration.

    Notes:
    - Keeps current project behavior: only `cockpit` camera uses non-zero mask.
    - Other cameras receive zero mask channel when mask conditioning is enabled.
    """

    policy: ACTPolicy
    qpos_mean: np.ndarray
    qpos_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    camera_names: list[str]
    device: torch.device
    use_mask_conditioning: bool = True
    mask_camera_name: str = "cockpit"
    num_queries: int = 1
    temporal_agg: bool = False
    _cached_actions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        stats_path: str,
        policy_config: Dict[str, Any],
        camera_names: list[str],
        device: Optional[torch.device] = None,
    ) -> "ACTPolicyRuntime":
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = _build_act_policy(policy_config)
        _ = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        policy.to(device)
        policy.eval()

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        return cls(
            policy=policy,
            qpos_mean=np.asarray(stats["qpos_mean"]),
            qpos_std=np.asarray(stats["qpos_std"]),
            action_mean=np.asarray(stats["action_mean"]),
            action_std=np.asarray(stats["action_std"]),
            camera_names=list(camera_names),
            device=device,
            use_mask_conditioning=bool(policy_config.get("use_mask_conditioning", False)),
            num_queries=int(policy_config.get("num_queries", 1)),
            temporal_agg=bool(policy_config.get("temporal_agg", False)),
        )

    def _pre_process_qpos(self, qpos: Any) -> torch.Tensor:
        q = np.asarray(qpos, dtype=np.float32)
        q_norm = (q - self.qpos_mean) / self.qpos_std
        return torch.from_numpy(q_norm).float().to(self.device).unsqueeze(0)

    def _post_process_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        action_np = action_tensor.squeeze(0).detach().cpu().numpy()
        return action_np * self.action_std + self.action_mean

    def _build_image_tensor(
        self,
        images_by_camera: Dict[str, Any],
        static_mask_by_camera: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        all_cam = []
        active_mask = self.mask_camera_name if self.mask_camera_name in self.camera_names else None

        for cam_name in self.camera_names:
            rgb = rearrange(images_by_camera[cam_name], "h w c -> c h w").astype(np.float32) / 255.0
            if self.use_mask_conditioning:
                if active_mask is not None and cam_name == active_mask:
                    if static_mask_by_camera is None or cam_name not in static_mask_by_camera:
                        # Runtime fallback: no oracle mask provided -> zero mask.
                        mask_2d = np.zeros(rgb.shape[1:], dtype=np.float32)
                    else:
                        mask_2d = np.asarray(static_mask_by_camera[cam_name], dtype=np.float32)
                else:
                    mask_2d = np.zeros(rgb.shape[1:], dtype=np.float32)
                rgb = np.concatenate([rgb, mask_2d[None, ...]], axis=0)
            all_cam.append(rgb)

        stacked = np.stack(all_cam, axis=0)
        return torch.from_numpy(stacked).float().to(self.device).unsqueeze(0)

    def reset_episode(self) -> None:
        self._cached_actions = None

    @torch.inference_mode()
    def step(self, step_input: PolicyStepInput) -> PolicyStepOutput:
        qpos_tensor = self._pre_process_qpos(step_input.qpos)
        image_tensor = self._build_image_tensor(
            images_by_camera=step_input.images_by_camera,
            static_mask_by_camera=step_input.static_mask_by_camera,
        )

        if self.temporal_agg:
            # Keep behavior explicit for now; can be implemented later if needed.
            all_actions = self.policy(qpos_tensor, image_tensor)
            raw_action = all_actions[:, 0]
        else:
            # Match ACT rollout logic:
            # - query policy every num_queries steps
            # - consume the predicted action chunk in order
            step_id = int(step_input.step_id)
            query_frequency = max(1, int(self.num_queries))
            if self._cached_actions is None or step_id % query_frequency == 0:
                self._cached_actions = self.policy(qpos_tensor, image_tensor)
            raw_action = self._cached_actions[:, step_id % query_frequency]

        action = self._post_process_action(raw_action)

        return PolicyStepOutput(
            action=action,
            meta={
                "target_id": int(step_input.brain_output.target_id),
                "used_mask_camera": self.mask_camera_name if self.mask_camera_name in self.camera_names else None,
                "step_id": int(step_input.step_id),
            },
        )
