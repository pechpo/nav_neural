"""CLI for multi-process data collection."""

from __future__ import annotations

import argparse
import multiprocessing as mp

from .eval.manager import DataManager, RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PointNav activations with multiprocessing")
    parser.add_argument(
        "--habitat-config",
        default="neural/configs/habitat.objectnav.hm3d.v2.yaml",
        help="Path to Habitat config YAML",
    )
    parser.add_argument(
        "--ckpt",
        default="data/pointnav_weights.pth",
        help="PointNav checkpoint path",
    )
    parser.add_argument("--out", default="neural/result.npz", help="Final output .npz path")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--num-dirs", type=int, default=12, help="Number of directions around the agent")
    parser.add_argument("--max-range", type=float, default=5.0, help="Max range in meters for obstacle rays")
    parser.add_argument("--min-range", type=float, default=0.0, help="Min range in meters for obstacle rays")
    parser.add_argument("--device", default="cuda", help="Torch device for PointNav policy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--skip-init", action="store_true", help="Skip planner init spin steps")
    parser.add_argument("--no-viz", action="store_true", help="Disable realtime visualization window")
    parser.add_argument("--no-video", action="store_true", help="Disable video saving")
    parser.add_argument("--video-dir", default="./videos", help="Directory for saved videos")
    parser.add_argument("--video-fps", type=int, default=10, help="FPS for saved videos")

    args = parser.parse_args()

    cfg = RunConfig(
        habitat_cfg=args.habitat_config,
        ckpt=args.ckpt,
        out=args.out,
        num_workers=int(args.num_workers),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        num_dirs=int(args.num_dirs),
        max_range=float(args.max_range),
        min_range=float(args.min_range),
        device=args.device,
        seed=int(args.seed),
        skip_init=bool(args.skip_init),
        realtime_viz=not bool(args.no_viz),
        save_video=not bool(args.no_video),
        video_dir=args.video_dir,
        video_fps=int(args.video_fps),
    )

    manager = DataManager(cfg)
    manager.run()


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
