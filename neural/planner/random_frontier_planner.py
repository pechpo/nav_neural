import random
from typing import Any, Dict, Optional

import numpy as np
from ..mapping.obstacle_map import ObstacleMap
from .gpu_client import GPUWorkerClient
from ..utils.geometry_utils import xyz_yaw_to_tf_matrix, rho_theta


class RandomFrontierPlanner:
    actions = ("move_forward", "turn_left", "turn_right", "stop")
    models = ["pointnav_rnn"]

    def __init__(self) -> None:
        # Two-phase navigation: initial spin then explore
        self._phase = "init"
        self._init_steps = 0
        self._max_init_steps = 12
        self._frontiers: np.ndarray = np.array([])
        self._goal_xy: np.ndarray = None
        self._curr_xy: np.ndarray = None
        self._curr_yaw: float = 0.0
        self._tf_camera_to_episodic: Optional[np.ndarray] = None
        self._should_reset_pointnav = False
        self._goal_history_steps = 0
        self._goal_rethink_interval = 30
        self._stuck_threshold = 3.0
        self._goal_path_since_rethink = 0.0
        self._last_agent_xy_for_rethink: Optional[np.ndarray] = None
        self._same_goal_thresh = 0.25

        # Obstacle map
        self._omap = ObstacleMap()

        # Runtime wiring (set by init_runtime)
        self._worker_id: int = None
        self._request_q = None
        self._reply_q = None
        self._proximity_threshold = 1.0
        self._ban_threshold = 0.5
        self._banned_points: np.ndarray = np.zeros((0, 2), dtype=np.float32)

        # GPU worker client (set in init_runtime)
        self._gpu: GPUWorkerClient
        self._pointnav_client = None

    @property
    def pointnav_client(self) -> Any:
        return self._pointnav_client

    def init_runtime(
        self,
        worker_id: Optional[int],
        request_q: Any = None,
        reply_q: Any = None,
        sim_sensors: Any = None,
        pointnav_client: Any = None,
    ) -> None:
        self._worker_id = worker_id
        self._request_q = request_q
        self._reply_q = reply_q
        if sim_sensors is None:
            raise ValueError("sim_sensors is required to initialize the planner.")
        camera_height = sim_sensors.rgb_sensor.position[1]
        min_depth = sim_sensors.depth_sensor.min_depth
        max_depth = sim_sensors.depth_sensor.max_depth
        camera_hfov_deg = sim_sensors.depth_sensor.hfov
        self._camera_height = float(camera_height)
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._camera_fov_rad = float(np.deg2rad(camera_hfov_deg))
        if pointnav_client is not None:
            self._pointnav_client = pointnav_client
            self._gpu = None
        else:
            if self._worker_id is None or self._request_q is None or self._reply_q is None:
                raise ValueError("GPU mode requires worker_id, request_q, and reply_q.")
            self._gpu = GPUWorkerClient(worker_id=self._worker_id, request_q=self._request_q, reply_q=self._reply_q)
            self._pointnav_client = self._gpu

    def reset(self) -> None:
        self._phase = "init"
        self._init_steps = 0
        self._frontiers = np.array([])
        self._goal_xy = None
        self._curr_xy = None
        self._curr_yaw = 0.0
        self._tf_camera_to_episodic = None
        self._omap.reset()
        # Ensure pointnav RNN state resets at episode start
        self._should_reset_pointnav = True
        self._goal_history_steps = 0
        self._goal_path_since_rethink = 0.0
        self._last_agent_xy_for_rethink = None
        # Clear banned points each episode
        self._banned_points = np.zeros((0, 2), dtype=np.float32)

    # ------------------------------ Ban Logic ------------------------------
    def is_banned(self, point_xy: Optional[np.ndarray]) -> bool:
        if point_xy is None:
            return False
        if not isinstance(point_xy, np.ndarray):
            point_xy = np.asarray(point_xy, dtype=np.float32)
        if point_xy.size != 2 or self._banned_points.size == 0:
            return False
        dists = np.linalg.norm(self._banned_points - point_xy.reshape(1, 2), axis=1)
        return bool(np.any(dists < float(self._ban_threshold)))

    def _ban_point(self, point_xy: Optional[np.ndarray]) -> None:
        if point_xy is None:
            return False
        if not isinstance(point_xy, np.ndarray):
            point_xy = np.asarray(point_xy, dtype=np.float32)
        if point_xy.size != 2:
            return False
        # Avoid adding duplicates very close to existing bans
        if self.is_banned(point_xy):
            return False
        if self._banned_points.size == 0:
            self._banned_points = point_xy.reshape(1, 2).astype(np.float32)
        else:
            self._banned_points = np.vstack([self._banned_points, point_xy.reshape(1, 2).astype(np.float32)])
        return True

    def _goal_is_occupied(self) -> bool:
        """Check whether the current goal lies on an obstacle in the obstacle map."""
        if self._goal_xy is None:
            return False
        return bool(self._omap.is_occupied_xy(self._goal_xy))

    def _update_mapping(self, depth: np.ndarray) -> None:
        # Ensure depth is HxW float32
        d = depth
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32, copy=False)
        h, w = d.shape[:2]
        fx = fy = float(w) / (2.0 * np.tan(float(self._camera_fov_rad) / 2.0))
        # Use transform precomputed at the start of act
        tf_camera_to_episodic = self._tf_camera_to_episodic

        # Update obstacle and explored maps; compute frontiers
        self._omap.update_map(
            d,
            tf_camera_to_episodic,
            float(self._min_depth),
            float(self._max_depth),
            fx,
            fy,
            float(self._camera_fov_rad),
            agent_yaw=self._curr_yaw,
        )
        self._omap.update_agent_traj(self._curr_xy, self._curr_yaw)
        # Pass banned frontier points to obstacle map for viz (draw 'X')
        self._omap.set_banned_frontiers(self._banned_points)
        # Filter frontiers against banned list
        fr = self._omap.frontiers
        if isinstance(fr, np.ndarray) and fr.size >= 2 and self._banned_points.size > 0:
            keep_mask = np.ones(fr.shape[0], dtype=bool)
            # Compute distances to each banned point and mask out close ones
            for b in self._banned_points:
                d = np.linalg.norm(fr - b.reshape(1, 2), axis=1)
                keep_mask &= d >= float(self._ban_threshold)
            self._frontiers = fr[keep_mask]
        else:
            self._frontiers = fr

    def _goal_is_same(self, prev_goal: Optional[np.ndarray], new_goal: Optional[np.ndarray]) -> bool:
        if prev_goal is None and new_goal is None:
            return True
        if prev_goal is None or new_goal is None:
            return False
        try:
            return bool(self._distance(prev_goal, new_goal) <= float(self._same_goal_thresh))
        except Exception:
            return False

    def _choose_goal(self) -> None:
        prev_goal = self._goal_xy.copy() if isinstance(self._goal_xy, np.ndarray) else None
        goal_xy = None
        if isinstance(self._frontiers, np.ndarray) and self._frontiers.size >= 2:
            candidates = self._frontiers
            if self._banned_points.size > 0:
                keep_mask = np.ones(candidates.shape[0], dtype=bool)
                for b in self._banned_points:
                    d = np.linalg.norm(candidates - b.reshape(1, 2), axis=1)
                    keep_mask &= d >= float(self._ban_threshold)
                candidates = candidates[keep_mask]
            if candidates.size >= 2:
                idx = random.randrange(candidates.shape[0])
                goal_xy = candidates[idx].astype(np.float32)
        self._goal_xy = goal_xy
        if not self._goal_is_same(prev_goal, goal_xy):
            self._goal_history_steps = 0
            self._goal_path_since_rethink = 0.0
            self._last_agent_xy_for_rethink = self._curr_xy.copy() if self._curr_xy is not None else None
        # Update visualization highlight for the selected frontier
        if hasattr(self._omap, "set_selected_frontier"):
            self._omap.set_selected_frontier(self._goal_xy)

    def _update_goal_progress(self) -> bool:
        if self._goal_xy is None or self._curr_xy is None:
            return False
        if self._last_agent_xy_for_rethink is None:
            self._last_agent_xy_for_rethink = self._curr_xy.copy()
        else:
            try:
                step_dist = float(np.linalg.norm(self._curr_xy - self._last_agent_xy_for_rethink))
            except Exception:
                step_dist = 0.0
            self._goal_path_since_rethink += step_dist
            self._last_agent_xy_for_rethink = self._curr_xy.copy()

        self._goal_history_steps += 1
        if self._goal_history_steps % int(self._goal_rethink_interval) == 0:
            if self._goal_path_since_rethink <= float(self._stuck_threshold):
                if self._ban_point(self._goal_xy):
                    print("PointNav stuck: banning current goal and replanning.")
                self._goal_xy = None
                if hasattr(self._omap, "set_selected_frontier"):
                    self._omap.set_selected_frontier(None)
                self._should_reset_pointnav = True
                self._goal_history_steps = 0
                self._goal_path_since_rethink = 0.0
                self._last_agent_xy_for_rethink = None
                return True
            self._goal_path_since_rethink = 0.0
            self._last_agent_xy_for_rethink = self._curr_xy.copy()
        return False
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _near_goal(self, threshold: float) -> bool:
        if self._goal_xy is None or self._curr_xy is None:
            raise ValueError("Cannot determine proximity to goal without active goal and current position.")
        d = self._distance(self._goal_xy, self._curr_xy)
        return d <= float(threshold)

    def _pointnav_step(self, depth: np.ndarray, goal_xy: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute pointgoal (rho, theta), run pointnav, and return action dict.

        Args:
            depth: Depth observation array (HxW or HxWx1).
            goal_xy: Goal position in world XY coordinates.
            extra: Optional extra key/values to merge into the returned dict
                   (e.g., det_seg for object planner visualizations).

        Returns:
            Dict with keys: action (str), map_viz (np.ndarray), and any extras.
        """
        rho, theta = rho_theta(self._curr_xy, self._curr_yaw, goal_xy)
        goal_vec = np.array([rho, theta], dtype=np.float32)
        # Capture and consume the current reset flag so that callers can
        # request a one-shot RNN reset. Any resets requested *inside* this
        # method (e.g., on STOP/ban) should apply to the *next* call.
        reset_flag = bool(self._should_reset_pointnav)
        if self._pointnav_client is None:
            raise RuntimeError("PointNav client is not initialized. Call init_runtime first.")
        act_id = self._pointnav_client.pointnav(depth, goal_vec, reset=reset_flag)
        if act_id == 0:
            # If the RNN outputs STOP, ban this goal to avoid revisiting
            new_ban = self._ban_point(goal_xy)
            if new_ban:
                print("PointNav STOP: banning current goal and overriding to MOVE_FORWARD.")
                # Update obstacle-map viz so the banned point is crossed out immediately
                if hasattr(self._omap, "set_banned_frontiers"):
                    self._omap.set_banned_frontiers(self._banned_points)
                # Clear current goal so the planner switches targets immediately
                try:
                    if self._goal_xy is not None and np.allclose(self._goal_xy.reshape(1, 2), goal_xy.reshape(1, 2)):
                        self._goal_xy = None
                        if hasattr(self._omap, "set_selected_frontier"):
                            self._omap.set_selected_frontier(None)
                        self._should_reset_pointnav = True
                except Exception:
                    # Be robust to any shape/type mismatch
                    self._goal_xy = None
                    if hasattr(self._omap, "set_selected_frontier"):
                        self._omap.set_selected_frontier(None)
                    self._should_reset_pointnav = True
            act_id = 1  # Override stop to move_forward to continue exploration
        # If a caller requested a reset for this step, clear it now so that
        # the reset is one-shot. Any resets set above (e.g., after STOP/ban)
        # will remain in effect for the next call.
        if reset_flag:
            self._should_reset_pointnav = False
        mapping = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}
        out: Dict[str, Any] = {
            "action": mapping.get(int(act_id), "move_forward"),
            "map_viz": self._omap.visualize(),
        }
        # Attach policy info useful for failure-cause analysis
        try:
            nav_goal_list = goal_xy.tolist() if hasattr(goal_xy, "tolist") else [float(goal_xy[0]), float(goal_xy[1])]
        except Exception:
            nav_goal_list = None
        out.update(
            {
                "nav_goal": nav_goal_list,
                "stop_called": out["action"] == "stop",
                "target_detected": False,  # frontier planner has no object detection
            }
        )
        if extra:
            out.update(extra)
        return out

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rgb = obs.get("rgb")  # unused here, but could be logged
        depth = obs.get("depth")
        gps = obs.get("gps")
        yaw = obs.get("yaw")

        # Precompute agent pose and transform once per step
        x, y = float(gps[0]), float(gps[1])
        camera_position = np.array([x, -y, self._camera_height], dtype=np.float32)
        self._curr_xy = camera_position[:2].copy()
        self._curr_yaw = float(yaw)
        self._tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, self._curr_yaw)

        # Update mapping with the precomputed transform
        self._update_mapping(depth)

        # If the current goal has been banned, immediately clear it so we switch targets
        if self._goal_xy is not None and self.is_banned(self._goal_xy):
            self._goal_xy = None
            # Clear highlight on the map as well
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._goal_history_steps = 0
            self._goal_path_since_rethink = 0.0
            self._last_agent_xy_for_rethink = None

        # If staircase protection (or other updates) mark the current goal as
        # an obstacle, drop and ban it so we replan instead of walking into it.
        if self._goal_xy is not None and self._goal_is_occupied():
            self._ban_point(self._goal_xy)
            self._goal_xy = None
            if hasattr(self._omap, "set_selected_frontier"):
                self._omap.set_selected_frontier(None)
            self._should_reset_pointnav = True
            self._goal_history_steps = 0
            self._goal_path_since_rethink = 0.0
            self._last_agent_xy_for_rethink = None

        # Initial spin for situational awareness
        if self._phase == "init" and self._init_steps < self._max_init_steps:
            self._init_steps += 1
            # Provide map visualization for UI
            out = {
                "action": "turn_left",
                "map_viz": self._omap.visualize(),
                "nav_goal": (self._goal_xy.tolist() if isinstance(self._goal_xy, np.ndarray) else None),
                "stop_called": False,
                "target_detected": False,
            }
            if self._init_steps >= self._max_init_steps:
                self._phase = "explore"
            return out

        # If we have an active goal and are close enough, request a reset of the
        # pointnav RNN state before pursuing the next goal to avoid state carryover.
        if self._goal_xy is not None:
            if self._update_goal_progress():
                self._choose_goal()
            if self._goal_xy is not None and self._near_goal(self._proximity_threshold):
                self._should_reset_pointnav = True
                self._choose_goal()
        else:
            # No active goal yet; just pick one without resetting.
            self._choose_goal()
        
        if self._goal_xy is None:
            return {
                "action": "stop",
                "map_viz": self._omap.visualize(),
                "nav_goal": None,
                "stop_called": True,
                "target_detected": False,
            }

        # Step pointnav towards goal
        return self._pointnav_step(depth, self._goal_xy)
