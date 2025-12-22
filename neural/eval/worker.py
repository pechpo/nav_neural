"""Worker process for data collection."""

from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from habitat.config.default import get_config
from habitat.core.env import Env

from ..io_utils import ensure_dir, save_json
from ..planner.random_frontier_planner import RandomFrontierPlanner
from ..utils import viz
try:
    from ..meausurements import exploration_map as _exploration_map  # noqa: F401
except Exception:
    _exploration_map = None  # type: ignore


@dataclass
class WorkerContext:
    worker_id: int
    habitat_cfg: str
    out_dir: str
    total_episodes: int
    max_steps: int
    num_dirs: int
    max_range: float
    min_range: float
    skip_init: bool
    realtime_viz: bool
    save_video: bool
    video_dir: str
    video_fps: int
    model_request_q: mp.Queue
    model_reply_q: mp.Queue
    episode_queue: mp.JoinableQueue
    example_dataset: Any
    stop_event: mp.Event


class DataWorker(mp.Process):
    def __init__(self, ctx: WorkerContext) -> None:
        super().__init__(daemon=False)
        self.ctx = ctx

    def run(self) -> None:
        cfg = get_config(config_path=self.ctx.habitat_cfg)
        env = Env(config=cfg, dataset=self.ctx.example_dataset)
        env._episode_from_iter_on_reset = False
        env._episode_force_changed = True

        planner = RandomFrontierPlanner()
        sim_sensors = env._config.simulator.agents.main_agent.sim_sensors
        planner.init_runtime(
            worker_id=self.ctx.worker_id,
            request_q=self.ctx.model_request_q,
            reply_q=self.ctx.model_reply_q,
            sim_sensors=sim_sensors,
        )
        client = planner.pointnav_client
        fov_rad = float(getattr(planner, "_camera_fov_rad", 0.0))
        angles = np.linspace(0.0, 2.0 * np.pi, int(self.ctx.num_dirs), endpoint=False)
        wrapped = (angles + np.pi) % (2.0 * np.pi) - np.pi
        visible_mask = np.abs(wrapped) <= (fov_rad * 0.5) if fov_rad > 0.0 else np.zeros_like(wrapped, dtype=bool)
        visible_indices = np.where(visible_mask)[0].tolist()
        invisible_indices = np.where(~visible_mask)[0].tolist()

        activations_hidden: List[np.ndarray] = []
        activations_rnn_output: List[np.ndarray] = []
        activations_visual_embed: List[np.ndarray] = []
        distances: List[np.ndarray] = []
        actions: List[int] = []
        poses: List[np.ndarray] = []
        episode_ids: List[int] = []

        show_viz = self.ctx.worker_id == 0 and self.ctx.realtime_viz
        save_video = self.ctx.worker_id == 0 and self.ctx.save_video
        window_name = "neural-worker-0"
        video_writer = None

        while not self.ctx.stop_event.is_set():
            try:
                ep_idx, ep = self.ctx.episode_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                env.current_episode = ep
                obs = env.reset()
                planner.reset()
                last_seen_call = getattr(client, "last_call_id", 0)
                if save_video:
                    video_path = str(Path(self.ctx.video_dir) / f"ep_{int(ep_idx):06d}.mp4")
                    video_writer = viz.VideoWriter(video_path, fps=self.ctx.video_fps)

                steps = 0
                while steps < int(self.ctx.max_steps):
                    if not env._task.is_episode_active:
                        break

                    rgb = obs.get("rgb")
                    depth = obs.get("depth")
                    gps = obs.get("gps")
                    yaw = obs.get("compass") if obs.get("compass") is not None else obs.get("heading")

                    action_dict = planner.act({"rgb": rgb, "depth": depth, "gps": gps, "yaw": yaw})
                    action = action_dict.get("action", "stop")

                    path_30 = None
                    if hasattr(planner, "_goal_path_since_rethink"):
                        try:
                            path_30 = float(getattr(planner, "_goal_path_since_rethink", 0.0))
                        except Exception:
                            path_30 = None

                    dists = None
                    if getattr(client, "last_call_id", 0) != last_seen_call:
                        last_seen_call = client.last_call_id
                        hidden = client.last_hidden
                        rnn_output = client.last_rnn_output
                        visual_embed = client.last_visual_embed
                        if (
                            hidden is not None
                            and rnn_output is not None
                            and visual_embed is not None
                            and not (self.ctx.skip_init and getattr(planner, "_phase", "") == "init")
                        ):
                            dists = planner._omap.raycast_obstacle_distances(
                                planner._curr_xy,
                                planner._curr_yaw,
                                num_dirs=self.ctx.num_dirs,
                                max_range_m=self.ctx.max_range,
                                min_range_m=self.ctx.min_range,
                            )
                            activations_hidden.append(hidden.astype(np.float32, copy=False))
                            activations_rnn_output.append(rnn_output.astype(np.float32, copy=False))
                            activations_visual_embed.append(visual_embed.astype(np.float32, copy=False))
                            distances.append(dists.astype(np.float32, copy=False))
                            actions.append(int(getattr(client, "last_action", -1)))
                            poses.append(
                                np.array(
                                    [planner._curr_xy[0], planner._curr_xy[1], float(planner._curr_yaw)],
                                    dtype=np.float32,
                                )
                            )
                            episode_ids.append(int(ep_idx))

                    if show_viz or save_video:
                        info = env.get_metrics()  # type: ignore[attr-defined]
                        dists_viz = dists
                        if dists_viz is None and planner._curr_xy is not None:
                            dists_viz = planner._omap.raycast_obstacle_distances(
                                planner._curr_xy,
                                planner._curr_yaw,
                                num_dirs=self.ctx.num_dirs,
                                max_range_m=self.ctx.max_range,
                                min_range_m=self.ctx.min_range,
                            )
                        rgb_vis = viz.rgb_to_bgr(rgb)
                        depth_vis = viz.depth_to_bgr(depth)
                        topdown = viz.render_gt_topdown(info)
                        obstacle = viz.render_obstacle_map(
                            planner._omap,
                            planner._curr_xy,
                            planner._curr_yaw,
                            ray_dists=dists_viz,
                            max_range_m=self.ctx.max_range,
                        )
                        grid = viz.make_grid([rgb_vis, topdown, depth_vis, obstacle])
                        grid = viz.overlay_status_bar(
                            grid,
                            steps=steps,
                            max_steps=self.ctx.max_steps,
                            extra_text=(
                                f"ep {int(ep_idx) + 1}/{int(self.ctx.total_episodes)}"
                                f"{'' if path_30 is None else f' | dist30: {path_30:.2f}m'}"
                            ),
                        )
                        if show_viz:
                            viz.show_frame(window_name, grid)
                        if save_video and video_writer is not None:
                            video_writer.write(grid)

                    step_in = {"action": action} if isinstance(action, str) else action
                    obs = env.step(step_in)
                    steps += 1
                    if action == "stop":
                        break
            finally:
                if video_writer is not None:
                    video_writer.close()
                    video_writer = None
                self.ctx.episode_queue.task_done()

        env.close()
        if show_viz:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        out_dir = ensure_dir(self.ctx.out_dir)
        out_path = Path(out_dir) / f"worker_{self.ctx.worker_id}.npz"
        act_hidden_arr = (
            np.stack(activations_hidden, axis=0) if activations_hidden else np.zeros((0, 0), dtype=np.float32)
        )
        act_rnn_output_arr = (
            np.stack(activations_rnn_output, axis=0) if activations_rnn_output else np.zeros((0, 0), dtype=np.float32)
        )
        act_visual_arr = (
            np.stack(activations_visual_embed, axis=0) if activations_visual_embed else np.zeros((0, 0), dtype=np.float32)
        )
        dist_arr = np.stack(distances, axis=0) if distances else np.zeros((0, 0), dtype=np.float32)

        np.savez_compressed(
            out_path,
            activations_hidden=act_hidden_arr,
            activations_rnn_output=act_rnn_output_arr,
            activations_visual_embed=act_visual_arr,
            distances=dist_arr,
            actions=np.asarray(actions, dtype=np.int32),
            poses=np.asarray(poses, dtype=np.float32),
            episode_ids=np.asarray(episode_ids, dtype=np.int32),
        )

        meta = {
            "worker_id": int(self.ctx.worker_id),
            "n_samples": int(act_hidden_arr.shape[0]),
            "activation_dim": int(act_hidden_arr.shape[1]) if act_hidden_arr.ndim == 2 else 0,
            "hidden_dim": int(act_hidden_arr.shape[1]) if act_hidden_arr.ndim == 2 else 0,
            "rnn_output_dim": int(act_rnn_output_arr.shape[1]) if act_rnn_output_arr.ndim == 2 else 0,
            "visual_embed_dim": int(act_visual_arr.shape[1]) if act_visual_arr.ndim == 2 else 0,
            "camera_fov_rad": float(fov_rad),
            "num_dirs": int(self.ctx.num_dirs),
            "visible_dir_mask": visible_mask.astype(bool).tolist(),
            "visible_dir_indices": visible_indices,
            "invisible_dir_indices": invisible_indices,
        }
        save_json(out_path.with_suffix(".json"), meta)
