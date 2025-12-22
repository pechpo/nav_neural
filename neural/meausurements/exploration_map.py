from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
from habitat import EmbodiedTask, registry
from habitat.core.embodied_task import Measure
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import HeadingSensor, NavigationEpisode, TopDownMap
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.utils.visualizations import maps
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from frontier_exploration.utils.general_utils import habitat_to_xyz


@registry.register_measure
class ExplorationMap(TopDownMap):
    """TopDownMap extension adding target bbox mask and feasibility info.

    Exposes additional fields on the measurement dict:
    - original_map: raw occupancy map before colorization
    - target_bboxes_mask: 2D mask of target object AABBs (same size as map)
    - upper_bound/lower_bound: sim bounds in meters (x,z)
    - grid_resolution: (H, W) of the topdown map grid
    - tf_episodic_to_global: 4x4 transform from episode frame to sim global
    - origin_in_pixel_coord: agent map coord at reset
    - is_feasible: episode feasible without stairs (bool)
    """

    def __init__(
        self,
        sim: HabitatSim,
        config,  # DictConfig
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config, *args, **kwargs)
        self._is_feasible: bool = True
        self._static_metrics: Dict[str, Any] = {}
        self._task = task

    def reset_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any) -> None:
        self._static_metrics = {}
        super().reset_metric(episode, *args, **kwargs)

        map_ori = self.get_original_map()
        self._draw_target_bbox_mask(episode)

        # Expose info for drawing 3D points on map
        lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
        episodic_start_yaw = HeadingSensor._quat_to_xy_heading(  # type: ignore
            None, quaternion_from_coeff(episode.start_rotation).inverse()
        )[0]
        # Convert Habitat world coordinates to the shared xyz frame used by
        # the planner/mappers and xyz_to_habitat. This keeps episodic/global
        # transforms consistent with how gps and point clouds are defined.
        x, y, z = habitat_to_xyz(np.array(episode.start_position))
        self._static_metrics["original_map"] = map_ori
        self._static_metrics["upper_bound"] = (upper_bound[0], upper_bound[2])
        self._static_metrics["lower_bound"] = (lower_bound[0], lower_bound[2])
        self._static_metrics["grid_resolution"] = self._metric["map"].shape[:2]
        self._static_metrics["tf_episodic_to_global"] = np.array(
            [
                [np.cos(episodic_start_yaw), -np.sin(episodic_start_yaw), 0, x],
                [np.sin(episodic_start_yaw), np.cos(episodic_start_yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):  # type: ignore[override]
        super().update_metric(episode, action, *args, **kwargs)
        new_map = self._metric["map"].copy()
        self._metric["map"] = new_map
        self._metric["is_feasible"] = self._is_feasible
        if "origin_in_pixel_coord" not in self._static_metrics:
            self._static_metrics["origin_in_pixel_coord"] = self._metric["agent_map_coord"]
        if not self._is_feasible:
            self._task._is_episode_active = False
        self._metric.update(self._static_metrics)

    def _draw_goals_view_points(self, episode):
        super()._draw_goals_view_points(episode)

        # Determine feasibility without stairs: check connected component containing start
        t_x, t_y = maps.to_grid(
            episode.start_position[2],
            episode.start_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        valid_with_viewpoints = self._top_down_map.copy()
        valid_with_viewpoints[valid_with_viewpoints == maps.MAP_VIEW_POINT_INDICATOR] = maps.MAP_VALID_POINT
        valid_with_viewpoints = cv2.dilate(valid_with_viewpoints, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(valid_with_viewpoints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_dist = np.inf
        best_idx = 0
        for idx, cnt in enumerate(contours):
            dist = cv2.pointPolygonTest(cnt, (t_y, t_x), True)
            if dist >= 0:
                best_idx = idx
                break
            elif abs(dist) < min_dist:
                min_dist = abs(dist)
                best_idx = idx

        best_cnt = contours[best_idx]
        mask = np.zeros_like(valid_with_viewpoints)
        mask = cv2.drawContours(mask, [best_cnt], 0, 1, -1)  # type: ignore
        masked_values = self._top_down_map[mask.astype(bool)]
        values = set(masked_values.tolist())
        self._is_feasible = maps.MAP_VALID_POINT in values and maps.MAP_VIEW_POINT_INDICATOR in values

    def _draw_target_bbox_mask(self, episode: NavigationEpisode):
        if not isinstance(episode, ObjectGoalNavEpisode):
            return
        bbox_mask = np.zeros_like(self._top_down_map)
        sem_scene = self._sim.semantic_annotations()
        for goal in episode.goals:
            object_id = goal.object_id  # type: ignore[attr-defined]
            center = sem_scene.objects[object_id].aabb.center
            x_len, _, z_len = sem_scene.objects[object_id].aabb.sizes / 2.0
            corners = [
                center + np.array([x, 0, z])
                for x, z in [(-x_len, -z_len), (x_len, z_len)]
                if self._is_on_same_floor(center[1])
            ]
            if not corners:
                continue
            map_corners = [
                maps.to_grid(
                    p[2], p[0], (self._top_down_map.shape[0], self._top_down_map.shape[1]), sim=self._sim
                )
                for p in corners
            ]
            (y1, x1), (y2, x2) = map_corners
            bbox_mask[y1:y2, x1:x2] = 1
        self._static_metrics["target_bboxes_mask"] = bbox_mask


@dataclass
class ExplorationMapMeasurementConfig(TopDownMapMeasurementConfig):
    type: str = ExplorationMap.__name__
    draw_waypoints: bool = True


# Register Hydra config node for Hydra-style defaults resolution
try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        package="habitat.task.measurements.exploration_map",
        group="habitat/task/measurements",
        name="exploration_map",
        node=ExplorationMapMeasurementConfig,
    )
except Exception:
    pass
