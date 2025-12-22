"""Minimal geometry utilities for PointNav + obstacle mapping."""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def get_rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    rotation_matrix = get_rotation_matrix(-curr_heading)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal
    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])
    return float(rho), float(theta)


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    x, y, z = xyz
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, x],
            [np.sin(yaw), np.cos(yaw), 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    original_shape = pts.shape
    d = original_shape[-1]
    flat = pts.reshape(-1, d)
    ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
    homogeneous = np.hstack((flat, ones))
    transformed = (transformation_matrix @ homogeneous.T).T
    out = transformed[:, :d] / transformed[:, d:]
    return out.reshape(original_shape)


def get_point_cloud(
    depth_image: np.ndarray,
    mask: Optional[np.ndarray],
    fx: float,
    fy: float,
    preserve_hw: bool = False,
) -> np.ndarray:
    if preserve_hw:
        v, u = np.indices(depth_image.shape)
        z = depth_image
        x = (u - depth_image.shape[1] // 2) * z / fx
        y = (v - depth_image.shape[0] // 2) * z / fy
        return np.stack((z, -x, -y), axis=-1)

    if mask is None:
        raise ValueError("mask must be provided when preserve_hw=False")
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    return np.stack((z, -x, -y), axis=-1)
