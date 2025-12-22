"""GPU manager for PointNav inference."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict

from ..models.pointnav_policy import PointNavPolicyClient
from ..utils.shared_memory import attach_ndarray


class GPUPointNavManager(mp.Process):
    def __init__(self, request_q: mp.Queue, reply_queues: Dict[int, mp.Queue], ckpt: str, device: str) -> None:
        super().__init__(daemon=False)
        self.request_q = request_q
        self.reply_queues = reply_queues
        self.ckpt = ckpt
        self.device = device
        self._pn_states: Dict[int, Dict[str, Any]] = {}

    def run(self) -> None:
        policy = PointNavPolicyClient(self.ckpt, device=self.device, deterministic=True)
        while True:
            msg = self.request_q.get()
            if msg is None:
                break
            if msg.get("type") == "shutdown" or msg.get("task") == "shutdown":
                break
            if msg.get("task") != "pointnav_rnn":
                continue

            req_id = msg.get("req_id")
            worker_id = int(msg.get("worker_id", -1))
            if worker_id not in self.reply_queues:
                continue

            depth_name = msg["depth_name"]
            depth_shape = tuple(msg["depth_shape"])
            depth_dtype = msg["depth_dtype"]
            goal_name = msg["goal_name"]
            goal_shape = tuple(msg["goal_shape"])
            goal_dtype = msg["goal_dtype"]
            reset = bool(msg.get("reset", False))

            depth_shm, depth_arr = attach_ndarray(depth_name, depth_shape, depth_dtype)
            goal_shm, goal_arr = attach_ndarray(goal_name, goal_shape, goal_dtype)
            try:
                state = None if reset else self._pn_states.get(worker_id)
                policy.set_state(state)
                action = policy.pointnav(depth_arr, goal_arr, reset=False)
                self._pn_states[worker_id] = policy.get_state()
                features = {
                    "hidden": policy.last_hidden,
                    "rnn_output": policy.last_rnn_output,
                    "visual_embed": policy.last_visual_embed,
                }
            finally:
                depth_shm.close()
                goal_shm.close()

            resp = {"req_id": req_id, "action": int(action), "features": features}
            self.reply_queues[worker_id].put(resp)
