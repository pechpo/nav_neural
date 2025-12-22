"""Lightweight obstacle map for frontier-based exploration."""

from __future__ import annotations

from typing import List, Optional, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from ..utils.geometry_utils import get_point_cloud, transform_points
from ..utils.img_utils import reorient_rescale_map
from .traj_visualizer import TrajectoryVisualizer


class ObstacleMap:
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float = 0.10,
        max_height: float = 0.88,
        agent_radius: float = 0.18,
        area_thresh: float = 1.5,
        hole_area_thresh: int = 100000,
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        self.pixels_per_meter = int(pixels_per_meter)
        self.size = int(size)
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        self._camera_positions: List[np.ndarray] = []
        self._last_camera_yaw: float = 0.0
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
        self._min_height = float(min_height)
        self._max_height = float(max_height)
        self._area_thresh_in_pixels = float(area_thresh) * (self.pixels_per_meter**2)
        self._hole_area_thresh = int(hole_area_thresh)

        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._frontiers_px: np.ndarray = np.array([])
        self.frontiers: np.ndarray = np.array([])
        self._selected_frontier_px: Optional[np.ndarray] = None
        self._banned_frontiers_px: np.ndarray = np.array([])
        self.additional_vis_info = {}

        kernel_size = int(self.pixels_per_meter * agent_radius * 2)
        kernel_size = kernel_size + (kernel_size % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        self._map.fill(0)
        self._navigable_map.fill(0)
        self._camera_positions = []
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
        self._last_camera_yaw = 0.0
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self._selected_frontier_px = None
        self._banned_frontiers_px = np.array([])
        self.additional_vis_info = {}

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        self._camera_positions.append(robot_xy)
        self._last_camera_yaw = float(robot_heading)

    def update_map(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        agent_yaw: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)

        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = d.copy()
                filled_depth[d == 0] = 1.0
            else:
                filled_depth = self._fill_small_holes(d, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            dense_points = get_point_cloud(scaled_depth, None, fx, fy, preserve_hw=True)
            dense_points_epi = transform_points(tf_camera_to_episodic, dense_points)
            sel = dense_points_epi[mask]
            if sel.size > 0:
                heights = sel[:, 2]
                ok = (heights >= self._min_height) & (heights <= self._max_height)
                obstacle_cloud = sel[ok]
                if obstacle_cloud.size > 0:
                    px = self._xy_to_px(obstacle_cloud[:, :2])
                    h, w = self._map.shape
                    valid = (
                        (px[:, 0] >= 0)
                        & (px[:, 0] < w)
                        & (px[:, 1] >= 0)
                        & (px[:, 1] < h)
                    )
                    pp = px[valid]
                    self._map[pp[:, 1], pp[:, 0]] = 1

            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]

        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-float(agent_yaw),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area[...] = new_area.astype(bool)

        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        if self.explored_area.size == 0:
            return np.array([])
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def _fill_small_holes(self, depth_img: np.ndarray, area_thresh: int) -> np.ndarray:
        binary_img = np.where(depth_img == 0, 1, 0).astype("uint8")
        contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filled_holes = np.zeros_like(binary_img)
        for cnt in contours:
            if cv2.contourArea(cnt) < area_thresh:
                cv2.drawContours(filled_holes, [cnt], 0, 1, -1)
        filled_depth_img = np.where(filled_holes == 1, 1, depth_img)
        return filled_depth_img

    def set_additional_vis_info(self, view_pos: np.ndarray = np.array([])) -> None:
        if len(view_pos):
            self.additional_vis_info["view_pos"] = self._xy_to_px(view_pos)

    def visualize(self) -> np.ndarray:
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (BGR: 200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        if self._selected_frontier_px is not None and len(self._selected_frontier_px) == 2:
            sf = tuple(int(i) for i in self._selected_frontier_px)
            cv2.circle(vis_img, sf, 7, (0, 0, 200), thickness=-1)

        if self.additional_vis_info.get("view_pos") is not None:
            for vp in self.additional_vis_info["view_pos"]:
                sf = tuple(int(i) for i in vp)
                cv2.circle(vis_img, sf, 5, (200, 0, 0), thickness=-1)

        if isinstance(self._banned_frontiers_px, np.ndarray) and self._banned_frontiers_px.size >= 2:
            for bp in self._banned_frontiers_px:
                bx, by = int(bp[0]), int(bp[1])
                size = 7
                color = (0, 0, 0)
                thickness = 2
                cv2.line(vis_img, (bx - size, by - size), (bx + size, by + size), color, thickness)
                cv2.line(vis_img, (bx - size, by + size), (bx + size, by - size), color, thickness)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        try:
            vis_img = reorient_rescale_map(vis_img)
        except Exception:
            pass

        return vis_img

    def is_occupied_xy(self, xy: Union[np.ndarray, list, tuple], radius_m: float = -0.5) -> bool:
        arr = np.asarray(xy, dtype=np.float32).reshape(1, 2)
        px = self._xy_to_px(arr)[0]
        x, y = int(px[0]), int(px[1])
        h, w = self._map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return True
        radius_p = int(abs(radius_m) * self.pixels_per_meter)
        t, b = max(y - radius_p, 0), min(y + radius_p, h)
        l, r = max(x - radius_p, 0), min(x + radius_p, w)
        func = np.any if radius_m > 0 else np.all
        return bool(func(self._map[t:b, l:r]).item())

    def set_selected_frontier(self, xy: Union[np.ndarray, None]) -> None:
        if xy is None:
            self._selected_frontier_px = None
            return
        arr = np.asarray(xy, dtype=np.float32).reshape(1, 2)
        self._selected_frontier_px = self._xy_to_px(arr)[0]

    def set_banned_frontiers(self, xy_points: Optional[np.ndarray]) -> None:
        if xy_points is None:
            self._banned_frontiers_px = np.array([])
            return
        arr = np.asarray(xy_points, dtype=np.float32)
        if arr.size == 0:
            self._banned_frontiers_px = np.array([])
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        self._banned_frontiers_px = self._xy_to_px(arr)

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        px = np.rint(pts[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)

    def _px_to_xy(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        px_copy = pts.copy()
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        xy = (px_copy - self._episode_pixel_origin) / float(self.pixels_per_meter)
        return xy[:, ::-1].astype(np.float32)

    def raycast_obstacle_distances(
        self,
        agent_xy: Union[np.ndarray, list, tuple],
        agent_yaw: float,
        num_dirs: int = 12,
        max_range_m: float = 5.0,
        min_range_m: float = 0.0,
    ) -> np.ndarray:
        if num_dirs <= 0:
            raise ValueError("num_dirs must be positive")
        agent_xy = np.asarray(agent_xy, dtype=np.float32).reshape(2)
        max_range_m = float(max_range_m)
        min_range_m = max(0.0, float(min_range_m))
        step_m = 1.0 / float(self.pixels_per_meter)
        steps = int(max_range_m / step_m)
        start_step = int(min_range_m / step_m)
        h, w = self._map.shape

        angles = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
        distances = np.full((num_dirs,), max_range_m, dtype=np.float32)

        for i, rel in enumerate(angles):
            ang = float(agent_yaw) + float(rel)
            direction = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            for s in range(start_step, steps):
                dist = (s + 1) * step_m
                point = agent_xy + direction * dist
                px = self._xy_to_px(point.reshape(1, 2))[0]
                x, y = int(px[0]), int(px[1])
                if x < 0 or x >= w or y < 0 or y >= h:
                    break
                if self._map[y, x]:
                    distances[i] = float(dist)
                    break

        return distances
