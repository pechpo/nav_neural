"""Visualization helpers for data collection."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def rgb_to_bgr(rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if rgb is None:
        return None
    img = rgb
    if img.dtype != np.uint8:
        vmax = float(np.nanmax(img)) if np.isfinite(img).any() else 1.0
        scale = 255.0 if vmax <= 1.0 else 1.0
        img = np.clip(img * scale, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def depth_to_bgr(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    d = depth
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32, copy=False)
    dmin = float(np.nanmin(d)) if np.isfinite(d).any() else 0.0
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 1.0
    if dmax <= dmin:
        dmax = dmin + 1.0
    norm = (d - dmin) / (dmax - dmin)
    img = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_grid(images: List[Optional[np.ndarray]], cell_size: int = 320) -> np.ndarray:
    tiles: List[np.ndarray] = []
    for img in images:
        if img is None:
            tiles.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
            continue
        h, w = img.shape[:2]
        if h != cell_size or w != cell_size:
            img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        tiles.append(img)
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


def overlay_status_bar(
    frame_bgr: np.ndarray,
    steps: int,
    max_steps: int,
    extra_text: Optional[str] = None,
) -> np.ndarray:
    """Append a status bar below the frame (no overlay on content)."""
    try:
        h, w = frame_bgr.shape[:2]
        bar_h = max(28, h // 24)
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (16, 16, 16)

        base = f"step: {int(steps)}/{int(max_steps)}"
        text = f"{extra_text} | {base}" if extra_text else base
        cv2.putText(
            bar,
            text,
            (12, int(bar_h * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([frame_bgr, bar])
    except Exception:
        return frame_bgr


def show_frame(window_name: str, frame: np.ndarray) -> None:
    try:
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    except Exception:
        return


class VideoWriter:
    def __init__(self, path: str, fps: int = 10) -> None:
        self.path = path
        self.fps = int(fps)
        self._writer: Optional[cv2.VideoWriter] = None
        self._size = None

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, float(self.fps), (w, h))
            self._size = (w, h)
        if self._size != (w, h):
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def render_gt_topdown(info: Optional[dict], max_size: int = 600) -> Optional[np.ndarray]:
    if not isinstance(info, dict):
        return None
    tdm = None
    if "exploration_map" in info:
        tdm = info.get("exploration_map")
    if tdm is None and "top_down_map" in info:
        tdm = info.get("top_down_map")
    if not isinstance(tdm, dict):
        return None
    top_down = tdm.get("map", None)
    if top_down is None:
        return None

    try:
        from habitat.utils.visualizations import maps

        fog = tdm.get("fog_of_war_mask", None)
        top_down_rgb = maps.colorize_topdown_map(top_down, fog)
    except Exception:
        if top_down.ndim == 2:
            top_down_rgb = np.repeat(top_down[:, :, None], 3, axis=2)
        else:
            top_down_rgb = top_down

    if top_down_rgb.dtype != np.uint8:
        top_down_rgb = np.clip(top_down_rgb, 0.0, 255.0).astype(np.uint8)
    top_down_bgr = cv2.cvtColor(top_down_rgb, cv2.COLOR_RGB2BGR)

    agent_coords = tdm.get("agent_map_coord", [])
    agent_angles = tdm.get("agent_angle", [])
    if isinstance(agent_coords, (tuple, list, np.ndarray)) and len(agent_coords) == 2 and not isinstance(
        agent_coords[0], (tuple, list, np.ndarray)
    ):
        agent_coords = [agent_coords]
        agent_angles = [tdm.get("agent_angle", 0.0)]

    try:
        from habitat.utils.visualizations import maps

        radius = max(1, min(top_down_bgr.shape[0:2]) // 32)
        for coord, ang in zip(agent_coords, agent_angles):
            top_down_bgr = maps.draw_agent(
                image=top_down_bgr,
                agent_center_coord=coord,
                agent_rotation=float(ang),
                agent_radius_px=radius,
            )
    except Exception:
        for coord in agent_coords:
            try:
                y, x = int(coord[0]), int(coord[1])
            except Exception:
                continue
            cv2.circle(top_down_bgr, (x, y), 4, (0, 255, 255), -1)

    return _resize_if_needed(top_down_bgr, max_size)


def render_topdown_map(
    omap: object,
    agent_xy: Optional[np.ndarray] = None,
    agent_yaw: Optional[float] = None,
    max_size: int = 600,
) -> np.ndarray:
    h, w = omap._map.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = 30
    explored = omap.explored_area.astype(bool)
    img[explored] = (160, 160, 160)
    img[omap._map] = (0, 0, 0)
    if agent_xy is not None:
        _draw_agent(omap, img, agent_xy, agent_yaw)
    return _resize_if_needed(img, max_size)


def render_obstacle_map(
    omap: object,
    agent_xy: Optional[np.ndarray] = None,
    agent_yaw: Optional[float] = None,
    ray_dists: Optional[np.ndarray] = None,
    max_range_m: Optional[float] = None,
    max_size: int = 600,
) -> np.ndarray:
    if hasattr(omap, "visualize"):
        try:
            vis = omap.visualize()
        except Exception:
            vis = None
        if vis is not None:
            return _resize_if_needed(vis, max_size)
    h, w = omap._map.shape
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    explored = getattr(omap, "explored_area", None)
    if explored is not None:
        img[np.asarray(explored, dtype=bool)] = (200, 255, 200)
    navigable = getattr(omap, "_navigable_map", None)
    if navigable is not None:
        img[np.asarray(navigable, dtype=bool) == 0] = (100, 100, 100)
    img[omap._map] = (0, 0, 0)
    for frontier in getattr(omap, "_frontiers_px", []):
        try:
            cv2.circle(img, tuple(int(i) for i in frontier), 5, (200, 0, 0), 2)
        except Exception:
            continue
    if agent_xy is not None:
        _draw_agent(omap, img, agent_xy, agent_yaw)
    if ray_dists is not None and agent_xy is not None and agent_yaw is not None:
        _draw_ray_hits(omap, img, agent_xy, agent_yaw, ray_dists, max_range_m=max_range_m)
    return _resize_if_needed(img, max_size)


def _draw_agent(omap: object, img: np.ndarray, agent_xy: np.ndarray, agent_yaw: Optional[float]) -> None:
    px = omap._xy_to_px(np.asarray(agent_xy, dtype=np.float32).reshape(1, 2))[0]
    cv2.circle(img, (int(px[0]), int(px[1])), 4, (0, 255, 255), -1)
    if agent_yaw is None:
        return
    heading = np.array([np.cos(agent_yaw), np.sin(agent_yaw)], dtype=np.float32)
    head_pt = agent_xy + heading * 0.5
    hp = omap._xy_to_px(head_pt.reshape(1, 2))[0]
    cv2.line(img, (int(px[0]), int(px[1])), (int(hp[0]), int(hp[1])), (0, 255, 255), 2)


def _draw_ray_hits(
    omap: object,
    img: np.ndarray,
    agent_xy: np.ndarray,
    agent_yaw: float,
    ray_dists: np.ndarray,
    max_range_m: Optional[float] = None,
) -> None:
    num_dirs = int(ray_dists.shape[0])
    colors = _direction_colors(num_dirs)
    max_range = float(max_range_m) if max_range_m is not None else float(np.max(ray_dists))
    step_m = 1.0 / float(omap.pixels_per_meter)
    for i in range(num_dirs):
        dist = float(ray_dists[i])
        if dist >= max_range - step_m * 0.5:
            continue
        ang = float(agent_yaw) + float(i) * (2.0 * np.pi / float(num_dirs))
        direction = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        point = agent_xy + direction * dist
        px = omap._xy_to_px(point.reshape(1, 2))[0]
        cv2.circle(img, (int(px[0]), int(px[1])), 3, colors[i], -1)


def _direction_colors(num_dirs: int) -> List[tuple]:
    if num_dirs <= 12:
        return [
            (255, 0, 0),
            (255, 128, 0),
            (255, 255, 0),
            (128, 255, 0),
            (0, 255, 0),
            (0, 255, 128),
            (0, 255, 255),
            (0, 128, 255),
            (0, 0, 255),
            (128, 0, 255),
            (255, 0, 255),
            (255, 0, 128),
        ][:num_dirs]
    colors = []
    for i in range(num_dirs):
        hue = int(179 * i / num_dirs)
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def _resize_if_needed(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = float(max_size) / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
