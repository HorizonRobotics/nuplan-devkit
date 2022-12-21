import numpy as np


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
