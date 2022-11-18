import math

import torch
import torch.nn.functional as F
from kornia.geometry.transform import get_rotation_matrix2d, warp_affine

from nuplan.planning.training.modeling.closed_loop_utilities.utils import \
    principal_value

DISTANCE_THRESH = 1.5
ANGLE_THRESH = 30 * math.pi / 180

# Ego coordinate system: x faces up, y faces left, yaw rotates counter-clockwise.
def repaint_raster(
    raster: torch.Tensor,
    pose_mats: torch.Tensor,
    target_pixel_size: float=0.5,
    dist_thresh: float=DISTANCE_THRESH,
    angle_thresh: float=ANGLE_THRESH
):
    assert raster.shape[-1] == raster.shape[-2], "Requires raster to be square."
    n = pose_mats.shape[0]
    output_size = raster.shape[-1]
    dist = torch.hypot(pose_mats[:, 0, 2], pose_mats[:, 1, 2])
    is_dist_in_bound = (dist <= dist_thresh)[:, None, None] 
    rot = torch.abs(
        -principal_value(
            torch.atan2(pose_mats[:, 1, 0], pose_mats[:, 0, 0])
        )
    )
    is_rot_in_bound = (rot <= angle_thresh)[:, None, None]
    is_in_bound = torch.logical_and(is_dist_in_bound, is_rot_in_bound)
    identity_matrix = torch.eye(3).repeat(n, 1, 1).type_as(pose_mats)
    bounded_pose_mats = torch.where(is_in_bound, pose_mats, identity_matrix)

    center = (torch.ones(n, 2) * output_size * 0.5).type_as(raster)
    rot_deg = (
        -principal_value(
            torch.atan2(bounded_pose_mats[:, 1, 0],
            bounded_pose_mats[:, 0, 0])
        ) * 180. / math.pi
    ).type_as(raster)
    scale = torch.ones(n, 2).type_as(raster)

    affine_mat = get_rotation_matrix2d(center, rot_deg, scale)
    affine_mat[:, 0, 2] = affine_mat[:, 0, 2] + bounded_pose_mats[:, 1, 2] / target_pixel_size
    affine_mat[:, 1, 2] = affine_mat[:, 1, 2] + bounded_pose_mats[:, 0, 2] / target_pixel_size

    output = warp_affine(raster, affine_mat, [output_size, output_size])
    output[:, 0, ...] = raster[:, 0, ...]

    return output, is_dist_in_bound, is_rot_in_bound, dist, rot
    # return output, is_in_bound
