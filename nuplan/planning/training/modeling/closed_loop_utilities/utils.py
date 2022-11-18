import math

import numpy as np
import torch


def principal_value(
    angle: torch.Tensor, min_: float = -math.pi
) -> torch.Tensor:
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).
    """
    assert torch.all(torch.isfinite(angle)), "angle is not finite"

    lhs = (angle - min_) % (2 * math.pi) + min_

    return lhs


def matrix_from_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """
    Convert prediction to pose matrices.

    :param prediction: batched prediction tensor [N, T, 3]
    :param time_point: if provided, should be an integer indicating which index to use to return the
        pose matrices.
    """
    assert prediction.ndim == 3
    n, t, _ = prediction.shape
    x = prediction[..., 0]
    y = prediction[..., 1]
    h = prediction[..., 2]
    row1 = torch.stack([torch.cos(h), -torch.sin(h), x], axis=-1)
    row2 = torch.stack([torch.sin(h), torch.cos(h), y], axis=-1)
    row3 = (
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        .repeat(n, t, 1)
        .to(prediction.device)
    )
    matrix = torch.stack([row1, row2, row3], dim=2)
    return matrix


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
