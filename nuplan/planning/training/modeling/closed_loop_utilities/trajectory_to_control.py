
import time
from typing import List

import numpy as np
import torch
from einops import einsum

from nuplan.common.actor_state.vehicle_parameters import \
    get_pacifica_parameters
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolverParameters, ILQRWarmStartParameters)
from nuplan.planning.training.modeling.closed_loop_utilities.utils.torch_util import \
    principal_value

INITIAL_CURVATURE_PENALTY = 1e-10


def _make_banded_difference_matrix(number_rows: int, device):
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].
    :param number_rows: The row dimension of the banded difference matrix (e.g. N-1 in the example above).
    :return: A banded difference matrix with shape (number_rows, number_rows+1).
    """
    banded_matrix = -1.0 * torch.eye(number_rows + 1, dtype=torch.float64, device=device)[:-1, :]
    for ind in range(len(banded_matrix)):
        banded_matrix[ind, ind + 1] = 1.0

    return banded_matrix

def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: torch.Tensor,
    velocity_profile: torch.Tensor,
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY,
):
    """
    Estimates initial curvature (curvature_0) and curvature rate ({curvature_rate_0, ...})
    using least squares with curvature rate regularization.
    :param heading_displacements: [rad] Angular deviations in heading occuring between timesteps.
    :param velocity_profile: [m/s] Estimated or actual velocities at the timesteps matching displacements.
    :param discretization_time: [s] Time discretization used for integration.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :param initial_curvature_penalty: A regularization parameter to handle zero initial speed.  Should be positive and small.
    :return: Least squares solution for initial curvature (curvature_0) and curvature rate profile
             (curvature_rate_0, ..., curvature_rate_{M-1}) for M heading displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert curvature_rate_penalty > 0.0, "Should have a positive curvature_rate_penalty."
    assert initial_curvature_penalty > 0.0, "Should have a positive initial_curvature_penalty."

    # Core problem: minimize_x ||y-Ax||_2
    num_batch, num_displacements = heading_displacements.shape
    device = heading_displacements.device

    y = heading_displacements.double()
    A = torch.ones((num_batch, num_displacements, num_displacements), dtype=torch.float64, device=device).tril_()
    A[:, :, 0] = velocity_profile * discretization_time
    A[:, 1:, 1:] *= velocity_profile[:, 1:, None] * discretization_time**2

    # Regularization on curvature rate.  We add a small but nonzero weight on initial curvature too.
    # This is since the corresponding row of the A matrix might be zero if initial speed is 0, leading to singularity.
    # We guarantee that Q is positive definite such that the minimizer of the least squares problem is unique.
    Q = curvature_rate_penalty * torch.eye(num_displacements, device=device, dtype=torch.double).tile(num_batch, 1, 1)
    Q[:, 0, 0] = initial_curvature_penalty

    # Compute regularized least squares solution.
    x = torch.linalg.pinv(A.transpose(1,2) @ A + Q) @ A.transpose(1,2) @ y[..., None]

    # Extract profile from solution.
    initial_curvature = x[:, 0]
    curvature_rate_profile = x[:, 1:, 0]

    return initial_curvature, curvature_rate_profile
    

def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: torch.Tensor, 
    heading_profile: torch.Tensor, 
    discretization_time: float, 
    jerk_penalty: float
):
    """
    Estimates initial velocity (v_0) and acceleration ({a_0, ...}) using least squares with jerk penalty regularization.
    :param xy_displacements: [m] Deviations in x and y occurring between M+1 poses, a M by 3 matrix.
    :param heading_profile: [rad] Headings associated to the starting timestamp for xy_displacements, a M-length vector.
    :param discretization_time: [s] Time discretization used for integration.
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :return: Least squares solution for initial velocity (v_0) and acceleration profile ({a_0, ..., a_M-1})
             for M displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert jerk_penalty > 0, "Should have a positive jerk_penalty."

    assert len(xy_displacements.shape) == 3, "Expect xy_displacements to be a 3-d tensor."
    assert xy_displacements.shape[2] == 2, "Expect xy_displacements to have 2 columns."

    num_displacements = xy_displacements.shape[1]  # aka M in the docstring

    assert heading_profile.shape[1] == num_displacements, "Expect the length of heading_profile to match that of xy_displacements."

    num_batch = xy_displacements.shape[0]
    device = xy_displacements.device

    y = xy_displacements.reshape(num_batch, -1)

    A = torch.zeros((num_batch, 2*num_displacements, num_displacements), dtype=torch.float64, device=device)
    A[:, 0::2, 0] = torch.cos(heading_profile) * discretization_time
    A[:, 1::2, 0] = torch.sin(heading_profile) * discretization_time
    for idx_timestep in range(num_displacements):
        if idx_timestep > 0:
            A[:, (2*idx_timestep)::2, 1:(1+idx_timestep)] = (torch.cos(heading_profile[:, idx_timestep:]) * discretization_time**2)[..., None]
            A[:, (2*idx_timestep+1)::2, 1:(1+idx_timestep)] = (torch.sin(heading_profile[:, idx_timestep:]) * discretization_time**2)[..., None]
    banded_matrix = _make_banded_difference_matrix(num_displacements - 2, device).tile((num_batch, 1, 1))
    R = torch.cat([
        torch.zeros((num_batch, banded_matrix.shape[1], 1), device=device),
        banded_matrix,
    ], dim=2)
    
    # Compute regularized least squares solution.
    x = torch.linalg.pinv(A.transpose(1, 2) @ A + jerk_penalty * R.transpose(1,2) @ R) @ A.transpose(1,2) @ y[..., None]
    # Extract profile from solution.
    initial_velocity = x[:, 0, 0:1]
    acceleration_profile = x[:, 1:, 0]

    return initial_velocity, acceleration_profile


def _generate_profile_from_initial_condition_and_derivatives(
    initial_condition: float, derivatives, discretization_time: float
):
    """
    Returns the corresponding profile (i.e. trajectory) given an initial condition and derivatives at
    multiple timesteps by integration.
    :param initial_condition: The value of the variable at the initial timestep.
    :param derivatives: The trajectory of time derivatives of the variable at timesteps 0,..., N-1.
    :param discretization_time: [s] Time discretization used for integration.
    :return: The trajectory of the variable at timesteps 0,..., N.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."

    profile = initial_condition + torch.cat(
        (torch.zeros_like(initial_condition), 
        torch.cumsum(derivatives * discretization_time, dim=1)), dim=1)

    return profile  # type: ignore

def _get_xy_heading_displacements_from_poses(poses):
    """
    Returns position and heading displacements given a pose trajectory.
    :param poses: <np.ndarray: num_batch, num_poses, 3> A trajectory of poses (x, y, heading).
    :return: Tuple of xy displacements with shape (num_batch, num_poses-1, 2) 
        and heading displacements with shape (num_batch, num_poses-1,).
    """
    assert len(poses.shape) == 3, "Expect a 3D matrix representing a trajectory of poses."
    assert poses.shape[1] > 1, "Cannot get displacements given an empty or single element pose trajectory."
    assert poses.shape[2] == 3, "Expect pose to have three elements (x, y, heading)."

    # Compute displacements that are used to complete the kinematic state and input.
    pose_differences = torch.diff(poses, axis=1)
    xy_displacements = pose_differences[:, :, :2]
    heading_displacements = principal_value(pose_differences[:, :, 2])

    return xy_displacements, heading_displacements

def _convert_curvature_profile_to_steering_profile(
    curvature_profile,
    discretization_time: float,
    wheel_base: float,
) :
    """
    Converts from a curvature profile to the corresponding steering profile.
    We assume a kinematic bicycle model where curvature = tan(steering_angle) / wheel_base.
    For simplicity, we just use finite differences to determine steering rate.
    :param curvature_profile: [rad] Curvature trajectory to convert.
    :param discretization_time: [s] Time discretization used for integration.
    :param wheel_base: [m] The vehicle wheelbase parameter required for conversion.
    :return: The [rad] steering angle and [rad/s] steering rate (derivative) profiles.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert wheel_base > 0.0, "The vehicle's wheelbase length must be positive."

    steering_angle_profile = torch.atan(wheel_base * curvature_profile)
    steering_rate_profile = torch.diff(steering_angle_profile) / discretization_time

    return steering_angle_profile, steering_rate_profile


def compute_steering_angle_feedback(
    pose_reference, pose_current, lookahead_distance: float, k_lateral_error: float
):
    """
    Given pose information, determines the steering angle feedback value to address initial tracking error.
    This is based on the feedback controller developed in Section 2.2 of the following paper:
    https://ddl.stanford.edu/publications/design-feedback-feedforward-steering-controller-accurate-path-tracking-and-stability
    :param pose_reference: <np.ndarray: num_batch, 3,> Contains the reference pose at the current timestep.
    :param pose_current: <np.ndarray: num_batch, 3,> Contains the actual pose at the current timestep.
    :param lookahead_distance: [m] Distance ahead for which we should estimate lateral error based on a linear fit.
    :param k_lateral_error: Feedback gain for lateral error used to determine steering angle feedback.
    :return: [rad] The steering angle feedback to apply.
    """
    assert pose_reference.shape[1] == 3, "We expect a single reference pose."
    assert pose_current.shape[1] == 3, "We expect a single current pose."

    assert lookahead_distance > 0.0, "Lookahead distance should be positive."
    assert k_lateral_error > 0.0, "Feedback gain for lateral error should be positive."

    x_reference, y_reference, heading_reference = pose_reference[:, 0], pose_reference[:, 1], pose_reference[:, 2]
    x_current, y_current, heading_current = pose_current[:, 0], pose_current[:, 1], pose_current[:, 2]

    x_error = x_current - x_reference
    y_error = y_current - y_reference
    heading_error = principal_value(heading_current - heading_reference)

    lateral_error = -x_error * torch.sin(heading_reference) + y_error * torch.cos(heading_reference)

    return (-k_lateral_error * (lateral_error + lookahead_distance * heading_error)).float()


def complete_kinematic_state_and_inputs_from_poses(
    discretization_time: float,
    wheel_base: float,
    poses,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) :
    """
    Main function for joint estimation of velocity, acceleration, steering angle, and steering rate given poses
    sampled at discretization_time and the vehicle wheelbase parameter for curvature -> steering angle conversion.
    One caveat is that we can only determine the first N-1 kinematic states and N-2 kinematic inputs given
    N-1 displacement/difference values, so we need to extrapolate to match the length of poses provided.
    This is handled by repeating the last input and extrapolating the motion model for the last state.
    :param discretization_time: [s] Time discretization used for integration.
    :param wheel_base: [m] The wheelbase length for the kinematic bicycle model being used.
    :param poses: <np.ndarray: num_batch, num_poses, 3> A trajectory of poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: kinematic_states (x, y, heading, velocity, steering_angle) and corresponding
            kinematic_inputs (acceleration, steering_rate).
    """
    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)

    # Compute initial velocity + acceleration least squares solution and extract results.
    # Note: If we have M displacements, we require the M associated heading values.
    #       Therefore, we exclude the last heading in the call below.
    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:, :-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    # Compute initial curvature + curvature rate least squares solution and extract results.  It relies on velocity fit.
    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    # Convert to steering angle given the wheelbase parameter.  At this point, we don't need to worry about curvature.
    steering_angle_profile, steering_rate_profile = _convert_curvature_profile_to_steering_profile(
        curvature_profile=curvature_profile,
        discretization_time=discretization_time,
        wheel_base=wheel_base,
    )

    # Extend input fits with a repeated element and extrapolate state fits to match length of poses.
    # This is since we fit with N-1 displacements but still have N poses at the end to deal with.
    acceleration_profile = torch.cat([acceleration_profile, acceleration_profile[:, -1].unsqueeze(1)], dim=1)
    steering_rate_profile = torch.cat([steering_rate_profile, steering_rate_profile[:, -1].unsqueeze(1)], dim=1)

    last_velocity = velocity_profile[:, -1] + acceleration_profile[:, -1] * discretization_time
    velocity_profile = torch.cat(
        [velocity_profile, last_velocity.unsqueeze(1)], dim=1
    )
    last_steering_angle = steering_angle_profile[:, -1] + steering_rate_profile[:, -1] * discretization_time
    steering_angle_profile = torch.cat(
        (steering_angle_profile, last_steering_angle.unsqueeze(1)), dim=1
    )

    # Collect completed state and input in matrices.
    kinematic_states = torch.cat((poses, velocity_profile.unsqueeze(2), steering_angle_profile.unsqueeze(2)), dim=2)

    kinematic_inputs = torch.stack((acceleration_profile, steering_rate_profile), dim=2)

    return kinematic_states, kinematic_inputs


class TrajectoryToControl:
    """Trajector to control variable based on iLQR solver implementation, see module docstring for details."""

    def __init__(
        self,
        solver_params: ILQRSolverParameters,
        warm_start_params: ILQRWarmStartParameters,
    ) -> None:
        """
        Initialize solver parameters.
        :param solver_params: Contains solver parameters for iLQR.
        :param warm_start_params: Contains warm start parameters for iLQR.
        """
        self._solver_params = solver_params
        self._warm_start_params = warm_start_params

        self._n_states = 5  # state dimension
        self._n_inputs = 2  # input dimension

        
        max_acceleration = self._solver_params.max_acceleration
        max_steering_angle_rate = self._solver_params.max_steering_angle_rate

        # Define input clip limits once to avoid recomputation in _clip_inputs.
        self._input_clip_min = (-max_acceleration, -max_steering_angle_rate)
        self._input_clip_max = (max_acceleration, max_steering_angle_rate)

        self._device = None

    def initialize(self, **kwargs):
        assert 'device' in kwargs
        self._device = kwargs['device']
        state_cost_diagonal_entries = self._solver_params.state_cost_diagonal_entries
        assert (
            len(state_cost_diagonal_entries) == self._n_states
        ), f"State cost matrix should have diagonal length {self._n_states}."
        state_cost_diagonal_entries = torch.tensor(state_cost_diagonal_entries, device=self._device)
        self._state_cost_matrix = torch.diag(state_cost_diagonal_entries)

        input_cost_diagonal_entries = self._solver_params.input_cost_diagonal_entries
        assert (
            len(input_cost_diagonal_entries) == self._n_inputs
        ), f"Input cost matrix should have diagonal length {self._n_inputs}."
        input_cost_diagonal_entries = torch.tensor(input_cost_diagonal_entries, device=self._device)
        self._input_cost_matrix = torch.diag(input_cost_diagonal_entries)

        state_trust_region_entries = self._solver_params.state_trust_region_entries
        assert (
            len(state_trust_region_entries) == self._n_states
        ), f"State trust region cost matrix should have diagonal length {self._n_states}."
        state_trust_region_entries = torch.tensor(state_trust_region_entries, device=self._device)
        self._state_trust_region_cost_matrix = torch.diag(state_trust_region_entries)

        input_trust_region_entries = self._solver_params.input_trust_region_entries
        assert (
            len(input_trust_region_entries) == self._n_inputs
        ), f"Input trust region cost matrix should have diagonal length {self._n_inputs}."
        input_trust_region_entries = torch.tensor(input_trust_region_entries, device=self._device)
        self._input_trust_region_cost_matrix = torch.diag(input_trust_region_entries)

    def solve(self, current_state, reference_trajectory):
        """
        Run the main iLQR loop used to try to find (locally) optimal inputs to track the reference trajectory.
        :param current_state: The initial state from which we apply inputs, z_0. [x, y, heading, velocity, steering angle]
        :param reference_trajectory: The state reference we'd like to track, inclusive of the initial timestep,
                                     z_{r,k} for k in {0, ..., N}.
        :return: A list of solution iterates after running the iLQR algorithm where the index is the iteration number.
        """
        # Check that state parameter has the right shape.
        assert current_state.shape[1] == self._n_states, "Incorrect state shape."

        # Check that reference trajectory parameter has the right shape.
        assert len(reference_trajectory.shape) == 3, "Reference trajectory should be a 3D matrix."
        num_batch, reference_trajectory_length, reference_trajectory_state_dimension = reference_trajectory.shape
        assert reference_trajectory_length > 1, "The reference trajectory should be at least two timesteps long."
        assert (
            reference_trajectory_state_dimension == self._n_states
        ), "The reference trajectory should have a matching state dimension."

        # List of ILQRSolution results where the index corresponds to the iteration of iLQR.
        solution_list: List[dict] = []

        # Get warm start input and state trajectory, as well as associated Jacobians.
        current_iterate = self._input_warm_start(current_state, reference_trajectory)

        # Main iLQR Loop.
        solve_start_time = time.perf_counter()
        for i in range(self._solver_params.max_ilqr_iterations):
            # Determine the cost and store the associated solution object.
            tracking_cost = self._compute_tracking_cost(
                iterate=current_iterate,
                reference_trajectory=reference_trajectory,
            )
            solution_list.append(
                dict(
                    input_trajectory=current_iterate['input_trajectory'],
                    state_trajectory=current_iterate['state_trajectory'],
                    tracking_cost=tracking_cost,
                )
            )

            # Determine the LQR optimal perturbations to apply.
            lqr_input_policy = self._run_lqr_backward_recursion(
                current_iterate=current_iterate,
                reference_trajectory=reference_trajectory,
            )

            # Apply the optimal perturbations to generate the next input trajectory iterate.
            input_trajectory_next = self._update_inputs_with_policy(
                current_iterate=current_iterate,
                lqr_input_policy=lqr_input_policy,
            )

            # Check for convergence/timeout and terminate early if so.
            # Else update the input_trajectory iterate and continue.
            input_trajectory_norm_difference = torch.linalg.norm(input_trajectory_next - current_iterate['input_trajectory'])

            current_iterate = self._run_forward_dynamics(current_state, input_trajectory_next)

            if input_trajectory_norm_difference < self._solver_params.convergence_threshold:
                break

            elapsed_time = time.perf_counter() - solve_start_time
            if (
                isinstance(self._solver_params.max_solve_time, float)
                and elapsed_time >= self._solver_params.max_solve_time
            ):
                break

        # Store the final iterate in the solution_dict.
        tracking_cost = self._compute_tracking_cost(
            iterate=current_iterate,
            reference_trajectory=reference_trajectory,
        )
        solution_list.append(
            dict(
                input_trajectory=current_iterate['input_trajectory'],
                state_trajectory=current_iterate['state_trajectory'],
                tracking_cost=tracking_cost,
            )
        )

        return solution_list

    
    def _input_warm_start(
        self,
        current_state, 
        reference_trajectory, 
        ):
        """
        Given a reference trajectory, we generate the warm start (initial guess) by inferring the inputs applied based
        on poses in the reference trajectory.
        :param current_state: The initial state from which we apply inputs. 
        :param reference_trajectory: The reference trajectory we are trying to follow.
        :return: The warm start control variable.
        """
        # poses = torch.cat([current_state[:, :3].unsqueeze(1), reference_trajectory[:, :, :3]], dim=1)
        reference_states_completed, reference_inputs_completed = complete_kinematic_state_and_inputs_from_poses(
            discretization_time=self._solver_params.discretization_time,
            wheel_base=self._solver_params.wheelbase,
            poses=reference_trajectory[:, :, :3],
            # poses=poses,
            jerk_penalty=self._warm_start_params.jerk_penalty_warm_start_fit,
            curvature_rate_penalty=self._warm_start_params.curvature_rate_penalty_warm_start_fit,
        )
        
        # We could just stop here and apply reference_inputs_completed (assuming it satisfies constraints).
        # This could work if current_state = reference_states_completed[0,:] - i.e. no initial tracking error.
        # We add feedback input terms for the first control input only to account for nonzero initial tracking error.
        velocity_current, steering_angle_current = current_state[:, 3], current_state[:, 4]
        velocity_reference, steering_angle_reference = reference_states_completed[:, 0, 3], reference_states_completed[:, 0, 4]

        acceleration_feedback = -self._warm_start_params.k_velocity_error_feedback * (
            velocity_current - velocity_reference
        )

        steering_angle_feedback = compute_steering_angle_feedback(
            pose_reference=current_state[:, :3],
            pose_current=reference_states_completed[:, 0, :3],
            lookahead_distance=self._warm_start_params.lookahead_distance_lateral_error,
            k_lateral_error=self._warm_start_params.k_lateral_error,
        )
        steering_angle_desired = steering_angle_feedback + steering_angle_reference
        steering_rate_feedback = -self._warm_start_params.k_steering_angle_error_feedback * (
            steering_angle_current - steering_angle_desired
        )

        reference_inputs_completed[:, 0, 0] += acceleration_feedback
        reference_inputs_completed[:, 0, 1] += steering_rate_feedback

        # We rerun dynamics with constraints applied to make sure we have a feasible warm start for iLQR.
        out = self._run_forward_dynamics(current_state, reference_inputs_completed)
        return out


    def _dynamics_and_jacobian(
        self,
        current_state, 
        current_input, 
        ) :
        """
        Propagates the state forward by one step and computes the corresponding state and input Jacobian matrices.
        We also impose all constraints here to ensure the current input and next state are always feasible.
        :param current_state: The current state z_k.
        :param current_input: The applied input u_k.
        :return: The next state z_{k+1}, (possibly modified) input u_k.
        """
        x, y, heading, velocity, steering_angle = current_state[:, 0], current_state[:, 1], current_state[:, 2], current_state[:, 3], current_state[:, 4]

        # Check steering angle is in expected range for valid Jacobian matrices.
        assert (
            torch.all(torch.abs(steering_angle) < np.pi / 2.0)
        ), f"The steering angle {steering_angle} is outside expected limits.  There is a singularity at delta = np.pi/2."

        # Input constraints: clip inputs within bounds and then use.
        current_input = self._clip_inputs(current_input)
        acceleration, steering_rate = current_input[:, 0], current_input[:, 1]

        # Euler integration of bicycle model.
        discretization_time = self._solver_params.discretization_time
        wheelbase = get_pacifica_parameters().wheel_base

        next_state = torch.clone(current_state)
        next_state[:, 0] += velocity * torch.cos(heading) * discretization_time
        next_state[:, 1] += velocity * torch.sin(heading) * discretization_time
        next_state[:, 2] += velocity * torch.tan(steering_angle) / wheelbase * discretization_time
        next_state[:, 3] += acceleration * discretization_time
        next_state[:, 4] += steering_rate * discretization_time

        # Constrain heading angle to lie within +/- pi.
        next_state[:, 2] = principal_value(next_state[:, 2])

        # State constraints: clip the steering_angle within bounds and update steering_rate accordingly.
        next_steering_angle = self._clip_steering_angle(next_state[:, 4])
        applied_steering_rate = (next_steering_angle - steering_angle) / discretization_time
        next_state[:, 4] = next_steering_angle
        current_input[:, 1] = applied_steering_rate

        # Now we construct and populate the state and input Jacobians.
        num_batch = current_input.shape[0]
        state_jacobian = torch.zeros([num_batch, self._n_states, self._n_states], device=self._device, dtype=torch.float64)
        idx = torch.arange(self._n_states)
        state_jacobian[:, idx, idx] = 1
        input_jacobian = torch.zeros([num_batch, self._n_states, self._n_inputs], device=self._device, dtype=torch.float64)

        # Set a nonzero velocity to handle issues when linearizing at (near) zero velocity.
        # This helps e.g. when the vehicle is stopped with zero steering angle and needs to accelerate/turn.
        # Without this, the A matrix will indicate steering has no impact on heading due to Euler discretization.
        # There will be a rank drop in the controllability matrix, so the discrete-time algebraic Riccati equation
        # may not have a solution (uncontrollable subspace) or it may not be unique.
        min_velocity_linearization = torch.tensor(self._solver_params.min_velocity_linearization, dtype=torch.float64, device=self._device)
        velocity = torch.where(
            torch.lt(torch.abs(velocity), min_velocity_linearization) & torch.gt(velocity, 0), min_velocity_linearization, velocity
        )
        velocity = torch.where(
            torch.lt(torch.abs(velocity), min_velocity_linearization) & torch.le(velocity, 0), -min_velocity_linearization, velocity
        )

        state_jacobian[:, 0, 2] = -velocity * torch.sin(heading) * discretization_time
        state_jacobian[:, 0, 3] = torch.cos(heading) * discretization_time

        state_jacobian[:, 1, 2] = velocity * torch.cos(heading) * discretization_time
        state_jacobian[:, 1, 3] = torch.sin(heading) * discretization_time

        state_jacobian[:, 2, 3] = torch.tan(steering_angle) / wheelbase * discretization_time
        state_jacobian[:, 2, 4] = velocity * discretization_time / (wheelbase * torch.cos(steering_angle) ** 2)

        input_jacobian[:, 3, 0] = discretization_time
        input_jacobian[:, 4, 1] = discretization_time

        return next_state, current_input, state_jacobian, input_jacobian


    def _run_forward_dynamics(
        self,
        current_state, 
        control_variables,
        ):
        """
        Compute states and corresponding state/input Jacobian matrices using forward dynamics.
        We additionally return the input since the dynamics may modify the input to ensure constraint satisfaction.
        :param current_state: The initial state from which we apply inputs.  Must be feasible given constraints.
        :param input_trajectory: The input trajectory applied to the model.  May be modified to ensure feasibility.
        :return: A feasible iterate after applying dynamics with state/input trajectories and Jacobian matrices.
        """
        num_batch, N = control_variables.shape[0], control_variables.shape[1]
        state_trajectory = torch.Tensor().new_full((num_batch, N+1, self._n_states), fill_value=float("nan"), dtype=torch.float64, device=self._device)
        final_input_trajectory = torch.Tensor().new_full(control_variables.shape, fill_value=float("nan"), dtype=torch.float64, device=self._device)
        state_jacobian_trajectory = torch.Tensor().new_full((num_batch, N, self._n_states, self._n_states), fill_value=float("nan"), dtype=torch.float64, device=self._device)
        final_input_jacobian_trajectory = torch.Tensor().new_full((num_batch, N, self._n_states, self._n_inputs), fill_value=float("nan"), dtype=torch.float64, device=self._device)

        state_trajectory[:, 0, :] = current_state

        for idx in range(N):
            u = control_variables[:, idx]
            state_next, final_input, state_jacobian, final_input_jacobian = self._dynamics_and_jacobian(
                    state_trajectory[:, idx], u
                )

            state_trajectory[:, idx + 1] = state_next
            final_input_trajectory[:, idx] = final_input
            state_jacobian_trajectory[:, idx] = state_jacobian
            final_input_jacobian_trajectory[:, idx] = final_input_jacobian
        
        iterate = dict(
            state_trajectory=state_trajectory,  # type: ignore
            input_trajectory=final_input_trajectory,  # type: ignore
            state_jacobian_trajectory=state_jacobian_trajectory,  # type: ignore
            input_jacobian_trajectory=final_input_jacobian_trajectory,  # type: ignore
        )

        return iterate

    
    def _compute_tracking_cost(
        self,
        iterate,
        reference_trajectory, 
        ):
        """
        Compute the trajectory tracking cost given a candidate solution.
        :param iterate: Contains the candidate state and input trajectory to evaluate.
        :param reference_trajectory: The desired state reference trajectory with same length as state_trajectory.
        :return: The tracking cost of the candidate state/input trajectory.
        """
        input_trajectory = iterate['input_trajectory']
        state_trajectory = iterate['state_trajectory']

        # assert state_trajectory.shape[1] == reference_trajectory.shape[1], "The state and reference trajectory should have the same length."

        # error_state_trajectory = state_trajectory[:, 1:, :] - reference_trajectory
        error_state_trajectory = state_trajectory - reference_trajectory
        error_state_trajectory[:, :, 2] = principal_value(error_state_trajectory[:, :, 2])

        input_cost = []
        for idx in range(input_trajectory.shape[1]):
            u = input_trajectory[:, idx]
            input_cost_idx = einsum(u, self._input_cost_matrix, 'b i, i i -> b i')
            input_cost_idx = einsum(input_cost_idx, u, 'b i, b i -> b')
            input_cost.append(torch.mean(input_cost_idx))

        state_cost = []
        for idx in range(error_state_trajectory.shape[1]):
            e = error_state_trajectory[:, idx]
            state_cost_idx = einsum(e, self._state_cost_matrix, 'b s, s s -> b s')
            state_cost_idx = einsum(state_cost_idx, e, 'b s, b s -> b')
            state_cost.append(torch.mean(state_cost_idx))

        input_cost = torch.tensor(input_cost, device=self._device, dtype=torch.float64)
        state_cost = torch.tensor(state_cost, device=self._device, dtype=torch.float64)

        cost = torch.sum(input_cost) + torch.sum(state_cost)
        return float(cost)


    def _clip_inputs(self, inputs):
        """
        Used to clip control inputs within constraints.
        :param: inputs: The control inputs with shape (self._n_i,) to clip.
        :return: Clipped version of the control inputs, unmodified if already within constraints.
        """
        assert inputs.shape[1] == 2, f"The inputs should be a 2D vector with 2 elements."
        _input_clip_min = torch.tensor(self._input_clip_min, device=self._device)
        _input_clip_max = torch.tensor(self._input_clip_max, device=self._device)

        return torch.clamp(inputs, _input_clip_min, _input_clip_max)  # type: ignore


    def _clip_steering_angle(self, steering_angle: float):
        """
        Used to clip the steering angle state within bounds.
        :param steering_angle: [rad] A steering angle (scalar) to clip.
        :return: [rad] The clipped steering angle.
        """
        _steering_angle = torch.minimum(torch.abs(steering_angle), torch.tensor(self._solver_params.max_steering_angle, device=self._device))
        steering_angle = torch.where(steering_angle >= 0, _steering_angle, -_steering_angle)
        return steering_angle


    def _run_lqr_backward_recursion(
        self,
        current_iterate,
        reference_trajectory
        ):
        """
        Computes the locally optimal affine state feedback policy by applying dynamic programming to linear perturbation
        dynamics about a specified linearization trajectory.  We include a trust region penalty as part of the cost.
        :param current_iterate: Contains all relevant linearization information needed to compute LQR policy.
        :param reference_trajectory: The desired state trajectory we are tracking.
        :return: An affine state feedback policy - state feedback matrices and feedforward inputs found using LQR.
        """
        state_trajectory = current_iterate['state_trajectory']
        input_trajectory = current_iterate['input_trajectory']
        state_jacobian_trajectory = current_iterate['state_jacobian_trajectory']
        input_jacobian_trajectory = current_iterate['input_jacobian_trajectory']

        input_cost_matrix = self._input_cost_matrix.unsqueeze(0).double()
        input_trust_region_cost_matrix = self._input_trust_region_cost_matrix.unsqueeze(0).double()
        state_cost_matrix = self._state_cost_matrix.unsqueeze(0).double()
        state_trust_region_cost_matrix = self._state_trust_region_cost_matrix.unsqueeze(0).double()

        # Check reference matches the expected shape.
        # assert reference_trajectory.shape[1] == state_trajectory.shape[1] - 1, "The reference trajectory has incorrect shape."
        # assert reference_trajectory.shape[1] == state_trajectory.shape[1], "The reference trajectory has incorrect shape."

        # Compute nominal error trajectory.
        # error_state_trajectory = state_trajectory[:, 1:, :] - reference_trajectory
        error_state_trajectory = state_trajectory - reference_trajectory
        error_state_trajectory[:, :, 2] = principal_value(error_state_trajectory[:, :, 2])

        # The value function has the form V_k(\Delta z_k) = \Delta z_k^T P_k \Delta z_k + 2 \rho_k^T \Delta z_k.
        # So p_current = P_k is related to the Hessian of the value function at the current timestep.
        # And rho_current = rho_k is part of the linear cost term in the value function at the current timestep.
        p_current = state_cost_matrix + state_trust_region_cost_matrix

        last_error_state_trajectory = error_state_trajectory[:, -1]
        rho_current = einsum(
            state_cost_matrix, last_error_state_trajectory, 
            'n_b n_s1 n_s, n_b n_s -> n_b n_s1')

        # The optimal LQR policy has the form \Delta u_k^* = K_k \Delta z_k + \kappa_k
        # We refer to K_k as state_feedback_matrix and \kappa_k as feedforward input in the code below.
        num_batch, N = input_trajectory.shape[0], input_trajectory.shape[1]
        state_feedback_matrices = float("nan") * torch.ones((num_batch, N, self._n_inputs, self._n_states), dtype=torch.float64, device=self._device)
        feedforward_inputs = float("nan") * torch.ones((num_batch, N, self._n_inputs), dtype=torch.float64, device=self._device)

        for i in reversed(range(N)):
            A = state_jacobian_trajectory[:, i]
            B = input_jacobian_trajectory[:, i]
            u = input_trajectory[:, i]
            error = error_state_trajectory[:, i]

            # Compute the optimal input policy for this timestep.
            # invertible since we checked input_cost / input_trust_region_cost are positive definite during creation.
            B_p = einsum(
                B, p_current,
                'n_b n_s1 n_i, n_b n_s1 n_s -> n_b n_i n_s')
            inverse_matrix = einsum(
                B_p, B,
                'n_b n_i1 n_s, n_b n_s n_i -> n_b n_i1 n_i'
            )
            inverse_matrix_term = -torch.linalg.inv(
                input_cost_matrix + input_trust_region_cost_matrix + inverse_matrix
            )

            state_feedback_matrix = einsum(
                inverse_matrix_term, B, p_current, A,  
                'n_b n_i1 n_i2, n_b n_s1 n_i2, n_b n_s1 n_s, n_b n_s n_s2 -> n_b n_i1 n_s2')

            input_u = einsum(
                input_cost_matrix, u, 
                'n_b n_i1 n_i2, n_b n_i2 -> n_b n_i1')
            B_rho = einsum(
                B, rho_current, 
                'n_b n_s n_i, n_b n_s -> n_b n_i')
            i_B = input_u + B_rho
            feedforward_input = einsum(
                inverse_matrix_term, i_B, 
                'n_b n_i1 n_i, n_b n_i -> n_b n_i1')

            # Compute the optimal value function for this timestep.
            B_s = einsum(
                B, state_feedback_matrix, 
                'n_b n_s1 n_i,  n_b n_i n_s2 -> n_b n_s1 n_s2')
            a_closed_loop = A + B_s

            a_p1 = einsum(
                a_closed_loop, p_current,
                'n_b n_s1 n_s2, n_b n_s1 n_s -> n_b n_s2 n_s'
            )
            a_p = einsum(
                a_p1, a_closed_loop,
                'n_b n_s1 n_s2, n_b n_s2 n_s3 -> n_b n_s1 n_s3'
            )

            cost_state_feedback_matrix = einsum(
                state_feedback_matrix, input_cost_matrix, state_feedback_matrix,
                'n_b n_i1 n_s1, n_b n_i1 n_i, n_b n_i n_s2  -> n_b n_s1 n_s2')

            trust_region_s_feedback_matrix = einsum(
                state_feedback_matrix, input_trust_region_cost_matrix, state_feedback_matrix,
                'n_b n_i1 n_s1, n_b n_i1 n_i2, n_b n_i2 n_s2 -> n_b n_s1 n_s2')
        
            p_prior = (
                state_cost_matrix 
                + state_trust_region_cost_matrix 
                + cost_state_feedback_matrix 
                + trust_region_s_feedback_matrix 
                + a_p
            )

            state_cost_error = einsum(
                state_cost_matrix, error, 
                'n_b n_s1 n_s2, n_b n_s2 -> n_b n_s1')

            f_u = feedforward_input + u
            state_feedback_matrix_cost = einsum(
                state_feedback_matrix, input_cost_matrix, f_u, 
                'n_b n_i1 n_s, n_b n_i1 n_i, n_b n_i -> n_b n_s')

            t_r = einsum(
                state_feedback_matrix, input_trust_region_cost_matrix, feedforward_input,
                'n_b n_i1 n_s, n_b n_i1 n_i, n_b n_i -> n_b n_s')
            
            a_p_f = einsum(
                a_closed_loop, p_current, B, feedforward_input,
                'n_b n_s1 n_s, n_b n_s1 n_s2, n_b n_s2 n_i, n_b n_i -> n_b n_s')

            a_r = einsum(
                a_closed_loop, rho_current,
                'n_b n_s1 n_s2, n_b n_s1 -> n_b n_s2'
            )

            rho_prior = (
                state_cost_error 
                + state_feedback_matrix_cost 
                + t_r 
                + a_p_f 
                + a_r
            )

            p_current = p_prior
            rho_current = rho_prior

            state_feedback_matrices[:, i] = state_feedback_matrix
            feedforward_inputs[:, i] = feedforward_input

        lqr_input_policy = dict(
            state_feedback_matrices=state_feedback_matrices,  # type: ignore
            feedforward_inputs=feedforward_inputs,  # type: ignore
        )

        return lqr_input_policy


    def _update_inputs_with_policy(
        self,
        current_iterate: dict,
        lqr_input_policy: dict,
        ):
        """
        Used to update an iterate of iLQR by applying a perturbation input policy for local cost improvement.
        :param current_iterate: Contains the state and input trajectory about which we linearized.
        :param lqr_input_policy: Contains the LQR policy to apply.
        :return: The next input trajectory found by applying the LQR policy.
        """
        state_trajectory = current_iterate['state_trajectory'] # [80, 17, 5]
        input_trajectory = current_iterate['input_trajectory'] # [80, 16, 2]

        # Trajectory of state perturbations while applying feedback policy.
        # Starts with zero as the initial states match exactly, only later states might vary.
        num_batch, N = input_trajectory.shape[0], input_trajectory.shape[1]
        delta_state_trajectory = torch.Tensor().new_full((num_batch, N+1, self._n_states), fill_value=float("nan"), dtype=torch.float64, device=self._device)
        delta_state_trajectory[:, 0, :] = 0.

        # This is the updated input trajectory we will return after applying the input perturbations.
        input_next_trajectory = torch.Tensor().new_full(input_trajectory.shape, fill_value=float("nan"), dtype=torch.float64, device=self._device)

        for idx in range(N):
            input_lin = input_trajectory[:, idx]
            state_lin = state_trajectory[:, idx] 
            state_lin_next = state_trajectory[:, idx+1]
            state_feedback_matrix = lqr_input_policy['state_feedback_matrices'][:, idx] 
            feedforward_input = lqr_input_policy['feedforward_inputs'][:, idx]
            # Compute locally optimal input perturbation.
            delta_state = delta_state_trajectory[:, idx]

            delta_input = einsum(
                state_feedback_matrix,
                delta_state,
                'n_b n_i n_s, n_b n_s -> n_b n_i'
            )
            delta_input = delta_input + feedforward_input

            # Apply state and input perturbation.
            input_perturbed = input_lin + delta_input
            state_perturbed = state_lin + delta_state
            state_perturbed[:, 2] = principal_value(state_perturbed[:, 2])

            # Run dynamics with perturbed state/inputs to get next state.
            # We get the actually applied input since it might have been clipped/modified to satisfy constraints.
            state_perturbed_next, input_perturbed, _, _ = self._dynamics_and_jacobian(state_perturbed, input_perturbed)

            # Compute next state perturbation given next state.
            delta_state_next = state_perturbed_next - state_lin_next
            delta_state_next[:, 2] = principal_value(delta_state_next[:, 2])

            delta_state_trajectory[:, idx + 1] = delta_state_next
            input_next_trajectory[:, idx] = input_perturbed

        assert ~torch.any(torch.isnan(input_next_trajectory)), "All next inputs should be valid float values."

        return input_next_trajectory  # type: ignore
