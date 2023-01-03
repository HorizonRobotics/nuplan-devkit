import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import torch

from nuplan.common.actor_state.ego_state import EgoState


@dataclass
class TrainingState:
    """Class for recording training status of each scenario, used in closed loop training."""

    state: EgoState
    last_prediction: Optional[torch.Tensor]
    current_iteration: int
    forced_reset_count: int = 0
    num_iter_without_reset: int = 0

    def __post_init__(self):
        if self.last_prediction is not None:
            assert self.last_prediction.shape[-1] == 3
        assert self.current_iteration >= 0

    @property
    def time_point(self):
        return self.state.time_point


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


def matrix_from_pose(prediction: torch.Tensor) -> torch.Tensor:
    """
    Convert prediction to pose matrices.

    Ref code: nuplan/common/geometry/convert.py#matrix_from_pose

    :param prediction: batched prediction tensor [N, T, 3]
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


def pose_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Generate trajectory params from matrix tensor.

    Ref code: nuplan/common/geometry/convert.py#pose_from_matrix

    :param matrix: [N, T, 3, 3] pose matrix tensor.
    """
    x = matrix[:, :, 0, 2]
    y = matrix[:, :, 1, 2]
    h = torch.atan2(matrix[:, :, 1, 0], matrix[:, :, 0, 0])
    return torch.stack([x, y, h], axis=-1)


class Interp1d(torch.autograd.Function):
    """Source code: https://github.com/aliutkus/torchinterp1d"""

    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
            assert len(vec.shape) <= 2, (
                "interp1d: all inputs must be " "at most 2-D."
            )
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, "All parameters must be on the same device."
        device = device[0]

        # Checking for the dimensions
        assert v["x"].shape[1] == v["y"].shape[1] and (
            v["x"].shape[0] == v["y"].shape[0]
            or v["x"].shape[0] == 1
            or v["y"].shape[0] == 1
        ), (
            "x and y must have the same number of columns, and either "
            "the same number of row or one of them having only one "
            "row."
        )

        reshaped_xnew = False
        if (
            (v["x"].shape[0] == 1)
            and (v["y"].shape[0] == 1)
            and (v["xnew"].shape[0] > 1)
        ):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v["xnew"].shape
            v["xnew"] = v["xnew"].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v["x"].shape[0], v["xnew"].shape[0])
        shape_ynew = (D, v["xnew"].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v["xnew"].shape[0] == 1:
            v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes
        torch.searchsorted(
            v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind
        )

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ["x", "y", "xnew"]:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [
                    None,
                ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat["slopes"] = is_flat["x"]
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
                eps + (v["x"][:, 1:] - v["x"][:, :-1])
            )

            # now build the linear interpolation
            ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
            ctx.saved_tensors[0],
            [i for i in inputs if i is not None],
            grad_out,
            retain_graph=True,
        )
        result = [
            None,
        ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


interp1d = Interp1d.apply
