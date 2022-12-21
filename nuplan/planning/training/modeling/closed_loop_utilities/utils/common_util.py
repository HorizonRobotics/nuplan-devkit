import numpy as np
import torch

from nuplan.planning.training.modeling.closed_loop_utilities.utils.numpy_util import \
    np_matrix_from_pose

THRESHOLD_LOW = [0, -1.0, 0.3]
THRESHOLD_HIGH = [1.0, 1.0, 0.3]


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
