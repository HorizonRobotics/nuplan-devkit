from __future__ import annotations

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import (
    AbstractMotionModel,
)
from nuplan.planning.simulation.controller.tracker.abstract_tracker import (
    AbstractTracker,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.closed_loop_utilities.abstract_training_controller import (
    AbstractTrainingController,
)
from nuplan.common.actor_state.state_representation import TimePoint

class TwoStageTrainingController(AbstractTrainingController):
    """
    This is a re-implementation of TwoStageController in simulation, to be used in training.
    """

    def __init__(
        self,
        step_time: float,
        future_horizon: float,
        train_tracker: AbstractTracker,
        train_motion_model: AbstractMotionModel,
    ):
        super().__init__(step_time, future_horizon)
        self._tracker = train_tracker
        self._motion_model = train_motion_model
        self.forced_reset_count = 0
        self.num_iter_without_reset = 0

    def initialize(self, state: EgoState, current_iteration: torch.Tensor):
        self.state = state
        self.current_iteration = current_iteration
        self._tracker.initialize()

    def set(self, state: EgoState, current_iteration: torch.Tensor):
        self.initialize(state, current_iteration)
        self.forced_reset_count += 1
        self.num_iter_without_reset = 0
        self.last_trajectory = None

    def update(self, timepoint: TimePoint) -> None:
        """Inherited, see superclass."""
        assert self.state is not None and self.current_iteration is not None

        states = transform_predictions_to_states(
            self.last_trajectory,
            self.state,
            self._future_horizon,
            self._step_interval,
        )
        trajectory = InterpolatedTrajectory(states)

        current_iteration = SimulationIteration(
            time_point=self.state.time_point, index=self.current_iteration.item()
        )

        sampling_time = timepoint - self.state.time_point

        next_iteration = SimulationIteration(
            time_point=timepoint, index=self.current_iteration.item() + 1
        )
        # Compute the dynamic state to propagate the model
        dynamic_state = self._tracker.track_trajectory(
            current_iteration, next_iteration, self.state, trajectory
        )

        self.state = self._motion_model.propagate_state(
            state=self.state,
            ideal_dynamic_state=dynamic_state,
            sampling_time=sampling_time,
        )
        self.current_iteration = self.current_iteration + 1
        self.num_iter_without_reset = self.num_iter_without_reset + 1