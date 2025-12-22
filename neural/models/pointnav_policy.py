"""PointNav policy wrapper with feature taps for data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete


@dataclass
class PointNavPolicyClient:
    ckpt_path: str
    device: Union[str, torch.device] = "cuda"
    deterministic: bool = True

    def __post_init__(self) -> None:
        from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
        from habitat_baselines.common.tensor_dict import TensorDict
        from habitat_baselines.rl.ppo.policy import PolicyActionData

        self._TensorDict = TensorDict  # type: ignore[attr-defined]
        self._PolicyActionData = PolicyActionData  # type: ignore[attr-defined]
        self._PointNavResNetPolicy = PointNavResNetPolicy

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        obs_space = SpaceDict(
            {
                "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Discrete(4)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.policy = self._PointNavResNetPolicy.from_config(ckpt["config"], obs_space, action_space)
        self.policy.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
        self.policy.to(self.device)
        self.policy.eval()

        hidden_size = int(getattr(self.policy.net, "recurrent_hidden_size", 512))
        self._hidden = torch.zeros(1, self.policy.net.num_recurrent_layers, hidden_size, device=self.device)
        self._prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self._step = 0
        self._call_id = 0

        self.last_action: Optional[int] = None
        self.last_hidden: Optional[np.ndarray] = None
        self.last_rnn_output: Optional[np.ndarray] = None
        self.last_visual_embed: Optional[np.ndarray] = None
        self.last_call_id: int = 0

    def reset(self) -> None:
        self._hidden = torch.zeros_like(self._hidden)
        self._prev_action = torch.zeros_like(self._prev_action)
        self._step = 0
        self._call_id = 0
        self.last_call_id = 0
        self.last_action = None
        self.last_hidden = None
        self.last_rnn_output = None
        self.last_visual_embed = None

    def get_state(self) -> Dict[str, Any]:
        return {
            "hidden": self._hidden.clone(),
            "prev_action": self._prev_action.clone(),
            "step": int(self._step),
        }

    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        if state is None:
            self.reset()
            return
        self._hidden = state["hidden"].to(self.device).clone()
        self._prev_action = state["prev_action"].to(self.device).clone()
        self._step = int(state.get("step", 0))

    def _prep_depth(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32, copy=False)
        denom = float(np.nanmax(d) or 5.0)
        denom = max(denom, 1e-6)
        d = np.clip(d / denom, 0.0, 1.0)
        if d.ndim == 2:
            d = d[..., None]
        if d.shape[0] != 224 or d.shape[1] != 224:
            import cv2

            d = cv2.resize(d, (224, 224), interpolation=cv2.INTER_AREA)
            if d.ndim == 2:
                d = d[..., None]
        return d

    @torch.inference_mode()
    def pointnav(
        self,
        depth: np.ndarray,
        goal_rho_theta: np.ndarray,
        reset: bool = False,
    ) -> int:
        if reset:
            self.reset()

        self._call_id += 1
        d = self._prep_depth(depth)
        obs = self._TensorDict(
            {
                "depth": torch.from_numpy(d).to(self.device).unsqueeze(0),
                "pointgoal_with_gps_compass": torch.from_numpy(goal_rho_theta.astype(np.float32))
                .to(self.device)
                .view(1, 2),
            }
        )

        masks = torch.tensor([[0 if self._step == 0 else 1]], dtype=torch.bool, device=self.device)
        self._step += 1

        # Feature taps (pre-action)
        try:
            out, _, aux = self.policy.net(obs, self._hidden, self._prev_action, masks)
            rnn_out = out.detach().float().cpu().numpy().reshape(-1)
            self.last_rnn_output = rnn_out
            vis = aux.get("perception_embed")
            if vis is not None:
                self.last_visual_embed = vis.detach().float().cpu().numpy().reshape(-1)
        except Exception:
            self.last_rnn_output = None
            self.last_visual_embed = None

        out = self.policy.act(obs, self._hidden, self._prev_action, masks, deterministic=self.deterministic)
        try:
            action_tensor = out.actions
            self._hidden = out.rnn_hidden_states
        except Exception:
            _, action_tensor, _, self._hidden = out

        self._prev_action = action_tensor.clone().to(self.device)
        self.last_action = int(action_tensor.view(-1)[0].item())
        self.last_hidden = self._hidden.detach().float().cpu().numpy().reshape(-1)
        self.last_call_id = self._call_id
        return self.last_action
