from typing import List

import numpy as np
import torch

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.controller.tracker.abstract_tracker import \
    AbstractTracker
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import \
    ILQRSolver
from nuplan.planning.training.modeling.closed_loop_utilities.utils.torch_util import (
    TrainingState, interp1d, matrix_from_pose, pose_from_matrix)


class ILQRTrainingTracker(AbstractTracker):
    """
    Tracker using an iLQR solver with a kinematic bicycle model. Batched version.
    """

    def __init__(self, n_horizon: int, ilqr_solver: ILQRSolver) -> None:
        """
        Initialize tracker parameters, primarily the iLQR solver.
        :param n_horizon: Maximum time horizon (number of discrete time steps) that we should plan ahead.
                          Please note the associated discretization_time is specified in the ilqr_solver.
        :param ilqr_solver: Solver used to compute inputs to apply.
        """
        assert n_horizon > 0, "The time horizon length should be positive."
        self._n_horizon = n_horizon

        self._ilqr_solver = ilqr_solver
        self._device = None
        self._initialized = False

    def initialize(self, **kwargs) -> None:
        """Inherited, see superclass."""
        assert (
            "device" in kwargs
        ), "Tracker requires a device to send tensor to."
        self._device = kwargs["device"]
        self._ilqr_solver.initialize(device=self._device)
        self._initialized = True

    def track_trajectory(
        self,
        initial_states: List[TrainingState],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        current_state = torch.tensor(
            [
                [
                    i.state.rear_axle.x,
                    i.state.rear_axle.y,
                    i.state.rear_axle.heading,
                    i.state.dynamic_car_state.rear_axle_velocity_2d.x,
                    i.state.tire_steering_angle,
                ]
                for i in initial_states
            ],
            device=self._device,
        )

        # Determine reference trajectory. This might be shorter than self._n_horizon states if near trajectory end.
        trajectory = self._prediction_to_absolute(initial_states)
        reference_trajectory = self._get_reference_trajectory(trajectory)
        # Run the iLQR solver to get the optimal input sequence to track the reference trajectory.
        solutions = self._ilqr_solver.solve(
            current_state, reference_trajectory[:, :]
        )
        optimal_inputs = solutions[-1]['input_trajectory']

        # Extract optimal input to apply at the current timestep.
        accel_cmd = optimal_inputs[:, 0, 0].cpu().numpy().tolist()
        steering_rate_cmd = optimal_inputs[:, 0, 1].cpu().numpy().tolist()

        return [
            DynamicCarState.build_from_rear_axle(
                rear_axle_to_center_dist=i.state.car_footprint.rear_axle_to_center_dist,
                rear_axle_velocity_2d=i.state.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=StateVector2D(accel_cmd_i, 0),
                tire_steering_rate=steering_rate_cmd_i,
            ) for i, accel_cmd_i, steering_rate_cmd_i in zip(initial_states, accel_cmd, steering_rate_cmd)
        ]

    def _get_reference_trajectory(
        self, trajectory: torch.Tensor, prediction_time_step: float = 0.5
    ) -> torch.Tensor:
        # current_iteration: N timepoints, 1 for each trajectory
        # trajectory: NxTx3 trajectory as poses
        """
        Determines reference trajectory, (z_{ref,k})_k=0^self._n_horizon.
        In case the query timestep exceeds the trajectory length, we return a smaller trajectory (z_{ref,k})_k=0^M,
        where M < self._n_horizon.  The shorter reference will then be handled downstream by the solver appropriately.
        :param current_iteration: Provides the current time from which we interpolate.
        :param trajectory: The full planned trajectory from which we perform state interpolation.
        :return a (M+1 or self._n_horizon+1) by self._n_states array.
        """
        # discretization_time = (
        #     self._ilqr_solver._solver_params.discretization_time
        # )

        batch, traj_length, _ = trajectory.shape
        interp_length = trajectory.shape[1]

        # trajectory = trajectory.transpose(1, 2).reshape(-1, traj_length)

        # time_delta_s = (
        #     torch.arange(0, self._n_horizon + 1) * discretization_time
        # )[None, :].repeat(trajectory.shape[0], 1)
        # prediction_time_seq = torch.arange(0.0, 8.5, prediction_time_step)[
        #     None, :
        # ].repeat(trajectory.shape[0], 1)

        # interp_length = time_delta_s.shape[-1]

        # interp_state = interp1d(prediction_time_seq, trajectory, time_delta_s)
        # interp_state = interp_state.reshape(
        #     batch, -1, interp_length
        # ).transpose(1, 2)

        pad = torch.zeros(batch, interp_length, 2).to(self._device)
        # interp_state = torch.cat([interp_state, pad], dim=-1)
        interp_state = torch.cat([trajectory, pad], dim=-1)

        return interp_state

    def _prediction_to_absolute(
        self, states: List[TrainingState]
    ) -> torch.Tensor:
        """Convert prediction poses (which is relative) to absolute poses.

        Ref code: nuplan/common/geometry/convert.py#relative_to_absolute_poses

        :param states: Tx3 tensor which represents x, y, heading in ego frame.
        :return same-sized tensor converted to absolute poses.
        """
        predictions = torch.stack([i.last_prediction for i in states], dim=0)
        traj_as_matrices = matrix_from_pose(predictions).to(self._device)
        state_as_matrices = torch.from_numpy(
            np.array([i.state.rear_axle.as_matrix() for i in states])
        )[:, None, :, :].to(
            self._device
        )
        abs_traj_as_matrices = state_as_matrices @ traj_as_matrices
        abs_traj = pose_from_matrix(abs_traj_as_matrices)

        return abs_traj
