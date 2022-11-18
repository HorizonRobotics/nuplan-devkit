from __future__ import annotations

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.planner.ml_planner.transform_utils import \
    transform_predictions_to_states
from nuplan.planning.simulation.trajectory.interpolated_trajectory import \
    InterpolatedTrajectory
from nuplan.planning.training.modeling.closed_loop_utilities.abstract_training_controller import \
    AbstractTrainingController


class PerfectTrackingEgoController(AbstractTrainingController):
    """Training tracker that implements perfect tracking strategy."""
    def __init__(self, step_time: float, future_horizon: float):
        super().__init__(step_time, future_horizon)
        self.forced_reset_count = 0
        self.num_iter_without_reset = 0

    def initialize(self, state: EgoState, current_iteration: torch.Tensor) -> None:
        """Inherited, see superclass."""
        self.state = state
        self.current_iteration = current_iteration

    def set(self, state: EgoState, current_iteration: torch.Tensor) -> None:
        """Inherited, see superclass."""
        self.initialize(state, current_iteration)
        self.forced_reset_count += 1
        self.num_iter_without_reset = 0
        self.last_trajectory = None

    def update(self, timepoint: TimePoint) -> None:
        """Inherited, see superclass."""
        assert self.last_trajectory is not None
        states = transform_predictions_to_states(
            self.last_trajectory,
            self.state,
            self._future_horizon,
            self._step_interval,
        )
        trajectory = InterpolatedTrajectory(states)
        self.state = trajectory.get_state_at_time(timepoint)
        assert self.state is not None
        self.current_iteration += 1
        self.num_iter_without_reset += 1
