from __future__ import annotations

import math

import theseus as th
import torch


class BatchedNonLinearSmoother(object):
    """
    Smoothing a set of xyh observations with a vehicle dynamics model.
    Original impl: nuplan/planning/training/data_augmentation/data_augmentation_util.py

    This is a batched version of ConstrainedNonLinearSmoother, offering faster speed.
    Does not guarantee the computed trajectory is the same as the original version.

    Example:
        # x_curr is [B, 4] tensor
        # reference_trajectory is [B, 17] tensor

        smoother = BatchedNonLinearSmoother(trajectory_len=16, dt=0.5)
        output, info = smoother.solve(x_curr, reference_trajectory)

        print(info.best_solution)
    """

    def __init__(
        self,
        trajectory_len: int,
        dt: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float,
        **kwargs
    ) -> None:
        """
        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        :param batch_size: batch size of the input.
        :param device: device on which all related tensors reside.
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self.nx = 4  # state dim
        self.nu = 2  # control dim

        self.objective = th.Objective(dtype=self.dtype)
        self._define_constants()
        self._create_decision_variables()
        self._create_parameters()
        self._set_dynamic_constraints()
        self._set_state_constraints()
        self._set_control_constraints()
        self._set_objective()
        self._create_layer()

    def solve(
        self, x_curr: torch.Tensor, reference_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Solve smoothing problem.

        :param x_curr: current position tensor [B, [x, y, yaw, speed]]
        :param reference_trajectory: reference traj [B, traj_len+1, [x, y, yaw]]
        """
        self._check_inputs(x_curr, reference_trajectory)

        damping = self.kwargs.get("damping", 0.1)
        smoother_inputs = {}
        # Set ref_traj and x_curr
        smoother_inputs["x_curr"] = x_curr
        for i in range(self.trajectory_len + 1):
            smoother_inputs[f"ref_traj_{i}"] = reference_trajectory[:, i, :]
        # Set initial guess
        for i in range(self.trajectory_len + 1):
            smoother_inputs[f"state_{i}"] = torch.cat(
                [
                    reference_trajectory[:, i, :],
                    x_curr[:, 3:],
                ],
                dim=-1,
            )
        # Forward layer
        with torch.no_grad():
            solved_values, info = self.smoother.forward(
                smoother_inputs,
                optimizer_kwargs={
                    "verbose": False,
                    "track_best_solution": True,
                    "damping": damping,
                },
            )

        # TODO: best_solution is on cpu, but final solution is on cuda. Use final
        # solution to avoid device transfer.
        # Concat best solution
        final_states = []
        for i in range(self.trajectory_len + 1):
            final_states.append(info.best_solution[f"state_{i}"])
        solved_states = torch.stack(final_states, dim=1)

        return solved_states

    def _create_decision_variables(self) -> None:
        """
        Define decision variables for the trajectory optimization.

        There are 2 types of decision variables: state and control.
        """
        # State trajectory (x, y, yaw, speed)
        self.state = []
        for i in range(self.trajectory_len + 1):
            self.state.append(
                th.Vector(
                    tensor=torch.zeros(
                        self.batch_size,
                        self.nx,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    name=f"state_{i}",
                )
            )

        # Control trajectory (curvature, accel)
        self.control = []
        for i in range(self.trajectory_len):
            self.control.append(
                th.Vector(
                    tensor=torch.zeros(
                        self.batch_size,
                        self.nu,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    name=f"control_{i}",
                )
            )

    def _create_parameters(self) -> None:
        """
        Define variables for input.

        There are 2 types of input: a reference trajectory and current position.
        """
        self.ref_traj = []
        for i in range(self.trajectory_len + 1):
            self.ref_traj.append(
                th.Variable(
                    tensor=torch.zeros(
                        self.batch_size,
                        3,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    name=f"ref_traj_{i}",
                )
            )

        self.x_curr = th.Variable(
            tensor=torch.zeros(
                self.batch_size, self.nx, dtype=self.dtype, device=self.device
            ),
            name="x_curr",
        )

    def _set_dynamic_constraints(self) -> None:
        """
        Set dynamic constraints. This is to ensure the traj continuity.
        """
        state = self.state
        control = self.control
        dt = self.dt

        def process(x_tensor, u_tensor) -> torch.Tensor:
            """
            Process for state propagation.

            :param x_tensor: [B, self.nx] shaped state tensor
            :param u_tensor: [B, self.nu] shaped control tensor
            """
            return torch.stack(
                [
                    x_tensor[:, 3] * torch.cos(x_tensor[:, 2]),
                    x_tensor[:, 3] * torch.sin(x_tensor[:, 2]),
                    x_tensor[:, 3] * u_tensor[:, 0],
                    u_tensor[:, 1],
                ],
                dim=-1,
            )

        def dynamic_error_fn(optim_vars, aux_vars) -> torch.Tensor:
            """
            Runge-Kutta 4 integration.

            :param states: [B, nx]
            :param control: [B, nu]
            """
            curr_state, curr_control, future_state = (
                i.tensor for i in optim_vars
            )
            k1 = process(curr_state, curr_control)
            k2 = process(curr_state + dt / 2 * k1, curr_control)
            k3 = process(curr_state + dt / 2 * k2, curr_control)
            k4 = process(curr_state + dt * k3, curr_control)
            next_state = curr_state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return future_state - next_state

        for k in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[
                        state[k],
                        control[k],
                        state[k + 1],
                    ],
                    err_fn=dynamic_error_fn,
                    dim=self.nx,
                    cost_weight=self.boundary_cost_weight,
                    name=f"dynamic_constraint_{k}",
                )
            )

    def _set_control_constraints(self) -> None:
        """Set control boundary constraints.

        NOTE: Theseus doesn't allow setting a hard limit, so we frame
        hard limits as cost-minimization problems. The cost function used is
            f(x) = min(0, x-upper_bound, -x+lower_bound)
        This ensures non-zero gradient when x is not in [lower_bound, upper_bound]
        range.
        """

        def curvature_constraint_fn(optim_vars, aux_vars) -> torch.Tensor:
            """Curvature constraint loss"""
            curvature_tensor = optim_vars[0].tensor[:, 0:1]
            curvature_limit = aux_vars[0].tensor
            curvature_constraint = self._bounded_loss(
                curvature_tensor, curvature_limit, -curvature_limit
            )
            return curvature_constraint

        def accel_constraint_fn(optim_vars, aux_vars) -> torch.Tensor:
            """Acceleration constraint loss"""
            accel_tensor = optim_vars[0].tensor[:, 1:]
            accel_limit = aux_vars[0].tensor
            accel_constraint = self._bounded_loss(
                accel_tensor, accel_limit, -accel_limit
            )
            return accel_constraint

        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.control[i]],
                    err_fn=curvature_constraint_fn,
                    dim=1,
                    cost_weight=self.boundary_cost_weight,
                    aux_vars=[self.curvature_limit],
                    name=f"curvature_constraint_{i}",
                )
            )
        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.control[i]],
                    err_fn=accel_constraint_fn,
                    dim=1,
                    cost_weight=self.boundary_cost_weight,
                    aux_vars=[self.accel_limit],
                    name=f"accel_constraint_{i}",
                )
            )

    def _set_state_constraints(self) -> None:
        """Set state boundary constraints."""

        def initial_pos_error(optim_vars, aux_vars) -> torch.Tensor:
            """Ensure the initial state is the same as 1st point of ref traj."""
            initial_pos_tensor = optim_vars[0].tensor  # [B, nx]
            ref_pos_tensor = aux_vars[0].tensor  # [B, nx]
            return initial_pos_tensor - ref_pos_tensor

        def speed_constraint_fn(optim_vars, aux_vars) -> torch.Tensor:
            """Set speed constraint."""
            curr_state = optim_vars[0].tensor
            max_speed = aux_vars[0].tensor
            speed = curr_state[:, 3:]
            return self._bounded_loss(speed, max_speed, self.zero)

        def yaw_rate_constraint_fn(optim_vars, aux_vars) -> torch.Tensor:
            """Set yaw rate constraint."""
            curr_state = optim_vars[0].tensor
            next_state = optim_vars[1].tensor
            max_yaw_rate = aux_vars[0].tensor
            yaw_rate = (next_state[:, 2:3] - curr_state[:, 2:3]) / self.dt
            return self._bounded_loss(yaw_rate, max_yaw_rate, -max_yaw_rate)

        def lateral_accel_fn(optim_vars, aux_vars) -> torch.Tensor:
            """Set lateral acceleration constraint."""
            curr_state = optim_vars[0].tensor
            curr_control = optim_vars[1].tensor
            max_lateral_accel = aux_vars[0].tensor
            lateral_accel = curr_state[:, 3:] ** 2 * curr_control[:, 0:1]
            return self._bounded_loss(
                lateral_accel, max_lateral_accel, -max_lateral_accel
            )

        self.objective.add(
            th.AutoDiffCostFunction(
                optim_vars=[self.state[0]],
                err_fn=initial_pos_error,
                dim=self.nx,
                cost_weight=self.boundary_cost_weight,
                aux_vars=[self.x_curr],
                name="initial_pos_constraint",
            )
        )

        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.state[i]],
                    err_fn=speed_constraint_fn,
                    dim=1,
                    cost_weight=self.boundary_cost_weight,
                    aux_vars=[self.max_speed],
                    name=f"speed_constraint_{i}",
                )
            )

        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.state[i], self.state[i + 1]],
                    err_fn=yaw_rate_constraint_fn,
                    dim=1,
                    cost_weight=self.boundary_cost_weight,
                    aux_vars=[self.max_yaw_rate],
                    name=f"yaw_rate_constraint_{i}",
                )
            )

        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.state[i], self.control[i]],
                    err_fn=lateral_accel_fn,
                    dim=1,
                    cost_weight=self.boundary_cost_weight,
                    aux_vars=[self.max_lateral_accel],
                    name=f"lateral_accel_constraint_{i}",
                )
            )

    def _set_objective(self) -> None:
        def cost_xy(optim_vars, aux_vars) -> torch.Tensor:
            """Compute per-state xy position loss."""
            state_xy = optim_vars[0].tensor[:, :2]
            ref_xy = aux_vars[0].tensor[:, :2]

            return state_xy - ref_xy

        def cost_yaw(optim_vars, aux_vars) -> torch.Tensor:
            """Compute per-state yaw loss."""
            state_yaw = optim_vars[0].tensor[:, 2:3]
            ref_yaw = aux_vars[0].tensor[:, 2:3]

            return state_yaw - ref_yaw

        def cost_rate(optim_vars, aux_vars) -> torch.Tensor:
            """Compute per-state curvature rate and jerk loss."""
            control_curr = optim_vars[0].tensor
            control_next = optim_vars[1].tensor
            curvature_rate_and_jerk = (control_next - control_curr) / self.dt
            return curvature_rate_and_jerk

        def cost_abs(optim_vars, aux_vars) -> torch.Tensor:
            """Compute per-state curvature and acceleration loss."""
            return optim_vars[0].tensor

        def cost_lat_accel(optim_vars, aux_vars) -> torch.Tensor:
            """Compute lateral acceleration loss."""
            speed_tensor = optim_vars[0].tensor[:, 3:]
            curvature_tensor = optim_vars[1].tensor[:, 0:1]
            return speed_tensor**2 * curvature_tensor

        def cost_terminal(optim_vars, aux_vars) -> torch.Tensor:
            """Compute terminal x/y/h loss."""
            last_state_tensor = optim_vars[0].tensor
            last_ref_tensor = aux_vars[0].tensor
            return last_ref_tensor - last_state_tensor[:, :3]

        # Cost XY
        for i in range(self.trajectory_len + 1):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.state[i]],
                    err_fn=cost_xy,
                    dim=2,
                    cost_weight=self.alpha_xy,
                    aux_vars=[self.ref_traj[i]],
                    name=f"cost_xy_{i}",
                )
            )

        # Cost yaw
        for i in range(self.trajectory_len + 1):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.state[i]],
                    err_fn=cost_yaw,
                    dim=1,
                    cost_weight=self.alpha_yaw,
                    aux_vars=[self.ref_traj[i]],
                    name=f"cost_yaw_{i}",
                )
            )

        # Cost terminal
        self.objective.add(
            th.AutoDiffCostFunction(
                optim_vars=[self.state[-1]],
                err_fn=cost_terminal,
                dim=3,
                cost_weight=self.alpha_terminal,
                aux_vars=[self.ref_traj[-1]],
                name="cost_terminal_xy",
            )
        )

        # Cost curvature rate and jerk
        for i in range(self.trajectory_len - 1):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.control[i], self.control[i + 1]],
                    err_fn=cost_rate,
                    dim=2,
                    cost_weight=self.alpha_rate,
                    name=f"cost_rate_{i}",
                )
            )

        # Cost curvature and accel
        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.control[i]],
                    err_fn=cost_abs,
                    dim=2,
                    cost_weight=self.alpha_abs,
                    name=f"cost_abs_{i}",
                )
            )

        # Cost lateral acceleration
        for i in range(self.trajectory_len):
            self.objective.add(
                th.AutoDiffCostFunction(
                    optim_vars=[self.state[i], self.control[i]],
                    err_fn=cost_lat_accel,
                    dim=1,
                    cost_weight=self.alpha_lat_accel,
                    name=f"cost_lat_accel_{i}",
                )
            )

    def _create_layer(self) -> None:
        """Create theseus layer."""
        max_iter = self.kwargs.get("max_iteration", 12)
        step_size = self.kwargs.get("step_size", 0.1)

        self.smoother = th.TheseusLayer(
            optimizer=th.LevenbergMarquardt(
                self.objective,
                linear_solver_cls=th.LUDenseSolver,
                max_iterations=max_iter,
                step_size=step_size,
            ),
        )
        self.smoother.to(device=self.device, dtype=self.dtype)

    def _define_constants(self):
        """Define a series of constants used as boundaries or weights."""

        # boundary_cost_weight
        boundary_cost_weight = float(self.kwargs.get("boundary_cost_weight", 10.0))

        # boundary weight
        self.boundary_cost_weight = th.ScaleCostWeight(
            torch.tensor(
                [[boundary_cost_weight]], dtype=self.dtype, device=self.device
            ),
            name="boundary_cost_weight",
        )

        # control constants
        # curvature_limit = 1.0 / 5.0 (unit: 1/m)
        # accel_limit = 4.0 (unit: m/s^2)
        self.curvature_limit = th.Variable(
            torch.tensor([[1.0 / 5.0]], dtype=self.dtype, device=self.device),
            name="curvature_limit",
        )
        self.accel_limit = th.Variable(
            torch.tensor([[4.0]], dtype=self.dtype, device=self.device),
            name="accel_limit",
        )

        # state constants
        # speed limit: [0, 35.0] (unit: m/s)
        # yaw rate: [-1.75, 1.75] (unit: rad/s)
        # lateral acceleration: [-4.0, 4.0] (unit: m/s^2)
        self.max_speed = th.Variable(
            torch.tensor([[35.0]], dtype=self.dtype, device=self.device),
            name="max_speed",
        )  # m/s
        self.max_yaw_rate = th.Variable(
            torch.tensor([[1.75]], dtype=self.dtype, device=self.device),
            name="max_yaw_rate",
        )  # rad/s
        self.max_lateral_accel = th.Variable(
            torch.tensor([[4.0]], dtype=self.dtype, device=self.device),
            name="max_lateral_accel",
        )  # m/s^2

        # objective weights
        self.alpha_xy = th.DiagonalCostWeight(
            torch.tensor([[1.0, 1.0]], dtype=self.dtype).sqrt(),
            name="alpha_xy",
        )
        self.alpha_yaw = th.ScaleCostWeight(
            torch.tensor([[0.1]], dtype=self.dtype, device=self.device).sqrt(),
            name="alpha_yaw",
        )
        self.alpha_rate = th.DiagonalCostWeight(
            torch.tensor(
                [[0.08, 0.08]], dtype=self.dtype, device=self.device
            ).sqrt(),
            name="alpha_rate",
        )
        self.alpha_abs = th.DiagonalCostWeight(
            torch.tensor(
                [[0.08, 0.08]], dtype=self.dtype, device=self.device
            ).sqrt(),
            name="alpha_abs",
        )
        self.alpha_lat_accel = th.ScaleCostWeight(
            torch.tensor([[0.06]], dtype=self.dtype).sqrt(),
            name="alpha_lat_accel",
        )
        self.alpha_terminal = th.DiagonalCostWeight(
            torch.tensor(
                [[1.0, 1.0, 40.0]], dtype=self.dtype, device=self.device
            ).sqrt()
            * math.sqrt(self.trajectory_len / 4.0),
            name="alpha_terminal",
        )

        # Utility constant
        self.zero = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    def _bounded_loss(
        self, target: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor
    ):
        """Imlement the bound constraint with non-zero gradient.

        f(x) = max(0., x - upper, -x+lower)
        """
        return torch.maximum(
            self.zero, torch.maximum(target - upper, -target + lower)
        )

    def _check_inputs(
        self, x_curr: torch.Tensor, reference_trajectory: torch.Tensor
    ) -> None:
        """Raise ValueError if inputs are not of proper sizes.

        :param x_curr: current state tensor of shape [N, self.nx]
        :param reference_trajectory: reference trajectory tensor of shape [N, self.trajectory_len+1, 3]
        :raises ValueError: if shape requirements don't meet.
        """
        if x_curr.shape[-1] != self.nx:
            raise ValueError(
                f"x_curr shape {x_curr.shape} must be equal to state dim {self.nx}"
            )

        if reference_trajectory.shape[1] != self.trajectory_len + 1:
            raise ValueError(
                f"reference traj length {reference_trajectory.shape[1]} must "
                f"be equal to {self.trajectory_len + 1}"
            )

        if (
            x_curr.dtype != self.dtype
            or reference_trajectory.dtype != self.dtype
        ):
            raise TypeError(
                f"Requires in put to be {self.dtype} type, got {x_curr.dtype} and "
                f"{reference_trajectory.dtype}."
            )

        if (
            x_curr.device != self.device
            or reference_trajectory.device != self.device
        ):
            raise ValueError(
                f"Requires input to be on device {self.device}, got {x_curr.device} "
                f"and {reference_trajectory.device}."
            )
