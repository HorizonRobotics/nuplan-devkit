import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint

THRESHOLD_LOW = [0, -1.0, 0.3]
THRESHOLD_HIGH = [1.0, 1.0, 0.3]


def get_relative_pose_matrices_to(
    gt_matrices: torch.Tensor, cl_matrices: torch.Tensor
) -> torch.Tensor:
    """Generate relative pose matrices realtive to ground truth.

    Args:
        gt_matrices (torch.Tensor): ground truth matrices.
        cl_matrices (torch.Tensor): closed loop or other matrices of the same shape.

    Returns:
        torch.Tensor: pose matrices relative to gt.
    """
    transform_origin = torch.linalg.inv(gt_matrices)
    transform_relative = transform_origin @ cl_matrices
    return transform_relative


def np_matrix_from_pose(pose: np.ndarray) -> np.ndarray:
    """Generate numpy pose matrices from pose array.

    Args:
        pose (np.ndarray): Nx3 array. The first 3 number in each row represents
            x, y, heading.

    Returns:
        np.ndarray: Nx3x3 matrices.
    """
    t, _ = pose.shape
    x = pose[..., 0]
    y = pose[..., 1]
    h = pose[..., 2]
    row1 = np.stack([np.cos(h), -np.sin(h), x], axis=-1)
    row2 = np.stack([np.sin(h), np.cos(h), y], axis=-1)
    row3 = np.array([[0.0, 0.0, 1.0]]).repeat(t, axis=0)
    matrix = np.stack([row1, row2, row3], axis=1)
    return matrix


def get_trajectories_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Generate trajectory params from pose matrices.

    The output of this function can be used as training gt.

    Args:
        matrix (np.ndarray): Nx3x3 numpy pose matices.

    Returns:
        np.ndarray: Nx3 array representing [x, y, h] of each traj point.
    """
    x = matrix[:, 0, 2]
    y = matrix[:, 1, 2]
    h = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    return np.stack([x, y, h], axis=-1)


def generate_random_perturb(N: int, device: torch.device) -> torch.Tensor:
    arrays = []
    for _ in range(N):
        if np.random.uniform(low=0.0, high=1.0) > 0.5:
            x_low, y_low, h_low = THRESHOLD_LOW
            x_high, y_high, h_high = THRESHOLD_HIGH
            x_perturb = np.random.uniform(low=x_low, high=x_high, size=(1,))
            y_perturb = np.random.uniform(low=y_low, high=y_high, size=(1,))
            h_perturb = np.random.uniform(low=h_low, high=h_high, size=(1,))
            sample_perturb = np.concat([x_perturb, y_perturb, h_perturb])
        else:
            sample_perturb = np.array([0.0, 0.0, 0.0])
        arrays.append(sample_perturb)
    array = np.stack(arrays, axis=-1)
    rel_pose_mats = np_matrix_from_pose(array)
    return torch.from_numpy(rel_pose_mats).to(device)
