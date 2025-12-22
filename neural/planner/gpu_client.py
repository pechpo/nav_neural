"""GPU worker client for PointNav only."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty
from typing import Any, Dict, Optional

import numpy as np

from ..utils.shared_memory import ShmArray


@dataclass
class GPUWorkerClient:
    worker_id: int
    request_q: Any
    reply_q: Any

    def __post_init__(self) -> None:
        self._req_id = 100000
        self._reply_timeout_sec = 30.0
        self._depth_slot = ShmArray()
        self._goal_slot = ShmArray()
        self.last_action: Optional[int] = None
        self.last_hidden: Optional[np.ndarray] = None
        self.last_rnn_output: Optional[np.ndarray] = None
        self.last_visual_embed: Optional[np.ndarray] = None
        self.last_call_id: int = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _await(self, req_id: int) -> Dict[str, Any]:
        while True:
            try:
                resp = self.reply_q.get(timeout=self._reply_timeout_sec)
            except Empty:
                continue
            if resp.get("req_id") == req_id:
                return resp

    def pointnav(self, depth: np.ndarray, goal_rho_theta: np.ndarray, reset: bool = False) -> int:
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)
        self._depth_slot.write(d)

        g = np.asarray(goal_rho_theta, dtype=np.float32)
        self._goal_slot.write(g)

        rid = self._next_id()
        ddesc = self._depth_slot.desc()
        gdesc = self._goal_slot.desc()
        msg = {
            "req_id": rid,
            "worker_id": int(self.worker_id),
            "task": "pointnav_rnn",
            "depth_name": ddesc["name"],
            "depth_shape": ddesc["shape"],
            "depth_dtype": ddesc["dtype"],
            "goal_name": gdesc["name"],
            "goal_shape": gdesc["shape"],
            "goal_dtype": gdesc["dtype"],
            "return_features": True,
        }
        if reset:
            msg["reset"] = True
        self.request_q.put(msg)
        resp = self._await(rid)
        self.last_call_id += 1
        if "features" in resp:
            feats = resp["features"] or {}
            self.last_hidden = feats.get("hidden")
            self.last_rnn_output = feats.get("rnn_output")
            self.last_visual_embed = feats.get("visual_embed")
        if "action" in resp:
            self.last_action = int(resp["action"])
            return self.last_action
        self.last_action = 0
        return 0

    def close(self) -> None:
        for slot in (self._depth_slot, self._goal_slot):
            if slot is not None:
                slot.close()
