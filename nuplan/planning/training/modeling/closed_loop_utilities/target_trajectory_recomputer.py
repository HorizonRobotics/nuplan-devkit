import numpy as np
import torch

from nuplan.planning.training.data_augmentation.data_augmentation_util import \
    ConstrainedNonlinearSmoother
from nuplan.planning.training.modeling.closed_loop_utilities.utils import (
    get_trajectories_from_matrix, np_matrix_from_pose, principal_value)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import \
    FeatureDataType

def _get_heading(sin_value: FeatureDataType, cos_value: FeatureDataType):
    """Get heading angle in radians.

    Args:
        sin_value (FeatureDataType): sine value of heading
        cos_value (FeatureDataType): cosine value of heading

    Returns:
        heading value in radians.
    """
    if isinstance(sin_value, torch.Tensor):
        return principal_value(torch.atan2(sin_value, cos_value))
    else:
        return np.arctan2(sin_value, cos_value)
class TargetTrajectoryRecomputer(object):
    """Recomputes trajectories based on perturbed location and a ref location."""
    def __init__(
        self,
        trajectory_len: int,
        time_interval: float,
    ):
        self._smoother = ConstrainedNonlinearSmoother(
            trajectory_len=trajectory_len,
            dt=time_interval
        )
        self._trajectory_len = trajectory_len
        self._time_interval = time_interval

    def reset(self):
        self._smoother = ConstrainedNonlinearSmoother(
            trajectory_len=self._trajectory_len,
            dt=self._time_interval
        )
    def recompute(
        self,
        original_target: FeatureDataType,
        pose_matrix: FeatureDataType,
        speed_magnitude: float
    ):
        ref_traj = [[0., 0., 0.]] + original_target.tolist()
        x = pose_matrix[0, 2].item()
        y = pose_matrix[1, 2].item()
        h = _get_heading(pose_matrix[1, 0], pose_matrix[0, 0]).item()
        x_curr = [x, y, h, speed_magnitude]
        self._smoother.set_reference_trajectory(
            x_curr=x_curr,
            reference_trajectory=ref_traj
        )
        try:
            sol = self._smoother.solve()
        except:
            return False, original_target
        if not sol.stats()['success']:
            return False, original_target
        else:
            ego_perturb = np.vstack(
                [
                    sol.value(self._smoother.position_x),
                    sol.value(self._smoother.position_y),
                    sol.value(self._smoother.yaw),
                ]
            )
            ego_perturb = ego_perturb.T[1:]
            ego_perturb = np_matrix_from_pose(ego_perturb)
            ego_perturb = np.linalg.inv(pose_matrix.cpu().numpy()) @ ego_perturb
            ego_perturb = get_trajectories_from_matrix(ego_perturb)
            new_target = torch.from_numpy(ego_perturb).type_as(original_target)
            return True, new_target
