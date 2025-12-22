"""Manager for multi-process data collection."""

from __future__ import annotations

import multiprocessing as mp
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from habitat.config.default import get_config
from habitat.datasets import make_dataset
from habitat.core.dataset import EpisodeIterator

from ..io_utils import ensure_dir, save_json
try:
    from ..meausurements import exploration_map as _exploration_map  # noqa: F401
except Exception:
    _exploration_map = None  # type: ignore
from .gpu_manager import GPUPointNavManager
from .worker import DataWorker, WorkerContext


@dataclass
class RunConfig:
    habitat_cfg: str
    ckpt: str
    out: str
    num_workers: int
    episodes: int
    max_steps: int
    num_dirs: int
    max_range: float
    min_range: float
    device: str
    seed: int
    skip_init: bool
    realtime_viz: bool
    save_video: bool
    video_dir: str
    video_fps: int


class DataManager:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.stop_event = mp.Event()

        out_path = Path(cfg.out)
        self.parts_dir = out_path.parent / f"{out_path.stem}_parts"
        ensure_dir(self.parts_dir)
        self.video_dir = Path(cfg.video_dir)
        if cfg.save_video:
            ensure_dir(self.video_dir)

        hcfg = get_config(config_path=cfg.habitat_cfg)
        dataset = make_dataset(id_dataset=hcfg.habitat.dataset.type, config=hcfg.habitat.dataset)
        self.example_dataset = copy(dataset)
        self.example_dataset.episodes = dataset.episodes[:1]
        self.total_eps = min(len(dataset.episodes), int(cfg.episodes))

        self.episode_queue: mp.JoinableQueue = mp.JoinableQueue()
        for idx, episode in enumerate(EpisodeIterator(dataset.episodes, shuffle=True, seed=int(cfg.seed))):
            if idx >= self.total_eps:
                break
            self.episode_queue.put((idx, episode))

        self.model_request_q: mp.Queue = mp.Queue()
        self.model_reply_qs = {wid: mp.Queue() for wid in range(int(cfg.num_workers))}

        self.workers: List[DataWorker] = []
        for worker_id in range(int(cfg.num_workers)):
            ctx = WorkerContext(
                worker_id=worker_id,
                habitat_cfg=cfg.habitat_cfg,
                out_dir=str(self.parts_dir),
                total_episodes=int(self.total_eps),
                max_steps=int(cfg.max_steps),
                num_dirs=int(cfg.num_dirs),
                max_range=float(cfg.max_range),
                min_range=float(cfg.min_range),
                skip_init=bool(cfg.skip_init),
                realtime_viz=bool(cfg.realtime_viz),
                save_video=bool(cfg.save_video),
                video_dir=str(self.video_dir),
                video_fps=int(cfg.video_fps),
                model_request_q=self.model_request_q,
                model_reply_q=self.model_reply_qs[worker_id],
                episode_queue=self.episode_queue,
                example_dataset=self.example_dataset,
                stop_event=self.stop_event,
            )
            self.workers.append(DataWorker(ctx))

        self.gpu_manager = GPUPointNavManager(
            request_q=self.model_request_q,
            reply_queues=self.model_reply_qs,
            ckpt=cfg.ckpt,
            device=cfg.device,
        )

    def run(self) -> dict:
        self.gpu_manager.start()
        for w in self.workers:
            w.start()

        try:
            self.episode_queue.join()
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self.stop_event.set()
            self.model_request_q.put({"type": "shutdown"})
            for w in self.workers:
                w.join()
            self.gpu_manager.join()

        return self._merge_parts()

    def _merge_parts(self) -> dict:
        acts_hidden: List[np.ndarray] = []
        acts_rnn_output: List[np.ndarray] = []
        acts_visual: List[np.ndarray] = []
        dists: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        episode_ids: List[np.ndarray] = []

        for part in sorted(self.parts_dir.glob("worker_*.npz")):
            data = np.load(part, allow_pickle=False)
            if "activations_hidden" in data:
                hidden = data["activations_hidden"]
            elif "activations" in data:
                hidden = data["activations"]
            else:
                continue
            if hidden.size == 0:
                continue
            acts_hidden.append(hidden)
            if "activations_rnn_output" in data:
                acts_rnn_output.append(data["activations_rnn_output"])
            if "activations_visual_embed" in data:
                acts_visual.append(data["activations_visual_embed"])
            dists.append(data["distances"])
            actions.append(data["actions"])
            poses.append(data["poses"])
            episode_ids.append(data["episode_ids"])

        out_path = Path(self.cfg.out)
        if acts_hidden:
            dist_arr = np.concatenate(dists, axis=0)
            actions_arr = np.concatenate(actions, axis=0)
            poses_arr = np.concatenate(poses, axis=0)
            episode_ids_arr = np.concatenate(episode_ids, axis=0)
            act_hidden_arr = np.concatenate(acts_hidden, axis=0) if acts_hidden else np.zeros((0, 0), dtype=np.float32)
            act_rnn_output_arr = (
                np.concatenate(acts_rnn_output, axis=0) if acts_rnn_output else np.zeros((0, 0), dtype=np.float32)
            )
            act_visual_arr = np.concatenate(acts_visual, axis=0) if acts_visual else np.zeros((0, 0), dtype=np.float32)
        else:
            dist_arr = np.zeros((0, 0), dtype=np.float32)
            actions_arr = np.zeros((0,), dtype=np.int32)
            poses_arr = np.zeros((0, 3), dtype=np.float32)
            episode_ids_arr = np.zeros((0,), dtype=np.int32)
            act_hidden_arr = np.zeros((0, 0), dtype=np.float32)
            act_rnn_output_arr = np.zeros((0, 0), dtype=np.float32)
            act_visual_arr = np.zeros((0, 0), dtype=np.float32)

        np.savez_compressed(
            out_path,
            activations_hidden=act_hidden_arr,
            activations_rnn_output=act_rnn_output_arr,
            activations_visual_embed=act_visual_arr,
            distances=dist_arr,
            actions=actions_arr,
            poses=poses_arr,
            episode_ids=episode_ids_arr,
        )

        hidden_dim = int(act_hidden_arr.shape[1]) if act_hidden_arr.ndim == 2 else 0
        rnn_output_dim = int(act_rnn_output_arr.shape[1]) if act_rnn_output_arr.ndim == 2 else 0
        visual_dim = int(act_visual_arr.shape[1]) if act_visual_arr.ndim == 2 else 0
        num_layers = None
        if hidden_dim > 0 and rnn_output_dim > 0 and hidden_dim % rnn_output_dim == 0:
            num_layers = int(hidden_dim // rnn_output_dim)

        visible_meta = {}
        worker_meta_path = next(iter(sorted(self.parts_dir.glob("worker_*.json"))), None)
        if worker_meta_path is not None:
            try:
                import json

                with worker_meta_path.open("r", encoding="utf-8") as f:
                    worker_meta = json.load(f)
                for key in (
                    "camera_fov_rad",
                    "num_dirs",
                    "visible_dir_mask",
                    "visible_dir_indices",
                    "invisible_dir_indices",
                ):
                    if key in worker_meta:
                        visible_meta[key] = worker_meta[key]
            except Exception:
                visible_meta = {}

        meta = {
            "habitat_config": str(Path(self.cfg.habitat_cfg).resolve()),
            "ckpt": str(Path(self.cfg.ckpt).resolve()),
            "episodes": int(self.cfg.episodes),
            "max_steps": int(self.cfg.max_steps),
            "num_dirs": int(self.cfg.num_dirs),
            "max_range": float(self.cfg.max_range),
            "min_range": float(self.cfg.min_range),
            "direction_names": [f"deg_{int(round(a))}" for a in np.linspace(0.0, 360.0, int(self.cfg.num_dirs), endpoint=False)],
            "n_samples": int(act_hidden_arr.shape[0]),
            "activation_dim": int(act_hidden_arr.shape[1]) if act_hidden_arr.ndim == 2 else 0,
            "hidden_dim": hidden_dim,
            "rnn_output_dim": rnn_output_dim,
            "visual_embed_dim": visual_dim,
            "rnn_num_layers": num_layers,
            "parts_dir": str(self.parts_dir.resolve()),
        }
        meta.update(visible_meta)
        save_json(out_path.with_suffix(".json"), meta)

        return {
            "n_samples": int(act_hidden_arr.shape[0]),
            "activation_dim": int(act_hidden_arr.shape[1]) if act_hidden_arr.ndim == 2 else 0,
        }
