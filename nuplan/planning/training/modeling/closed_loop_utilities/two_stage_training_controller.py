from __future__ import annotations

from typing import List

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import \
    AbstractMotionModel
from nuplan.planning.simulation.controller.tracker.abstract_tracker import \
    AbstractTracker
from nuplan.planning.simulation.planner.ml_planner.transform_utils import \
    transform_predictions_to_states
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import \
    SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import \
    InterpolatedTrajectory
from nuplan.planning.training.modeling.closed_loop_utilities.abstract_training_controller import \
    AbstractTrainingController
from nuplan.planning.training.modeling.closed_loop_utilities.ilqr_training_tracker import \
    ILQRTrainingTracker
from nuplan.planning.training.modeling.closed_loop_utilities.kinematic_bicycle_batched import \
    BatchedKinematicBicycleModel
from nuplan.planning.training.modeling.closed_loop_utilities.utils.torch_util import \
    TrainingState


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
            time_point=self.state.time_point,
            index=self.current_iteration.item(),
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


class TwoStageTrainingControllerBatched(AbstractTrainingController):
    def __init__(
        self,
        step_time: float,
        future_horizon: float,
        train_tracker: ILQRTrainingTracker,
        train_motion_model: BatchedKinematicBicycleModel,
    ):
        super().__init__(step_time, future_horizon)
        assert isinstance(
            train_tracker, ILQRTrainingTracker
        ), "Batched training controller only supports ILQRTrainingTracker now."
        assert isinstance(
            train_motion_model, BatchedKinematicBicycleModel
        ), "Batched training controller only supports BatchedKinematicBicycleModel now."
        self._tracker = train_tracker
        self._motion_model = train_motion_model
        self._device = None

    def initialize(self, **kwargs) -> None:
        """Inherited, see superclass."""
        assert (
            "device" in kwargs
        ), "Device needs to be provided during initialization."
        print(f"====initialize batched controller on {kwargs['device']}====")
        self._device = kwargs["device"]
        self._tracker.initialize(**kwargs)
        self._motion_model.initialize(**kwargs)

    def set(self):
        pass

    def update(
        self, training_states: List[TrainingState], timepoints: List[TimePoint]
    ) -> List[TrainingState]:
        """Inherited, see superclass."""
        assert len(training_states) == len(timepoints)

        dynamic_states = self._tracker.track_trajectory(training_states)
        sampling_time = [
            next_t - curr_t
            for next_t, curr_t in zip(
                timepoints,
                [i.time_point for i in training_states],
            )
        ]
        states = self._motion_model.propagate_state(
            state=[i.state for i in training_states],
            ideal_dynamic_state=dynamic_states,
            sampling_time=sampling_time,
        )

        for training_state, state_i in zip(training_states, states):
            training_state.current_iteration += 1
            training_state.state = state_i
            training_state.last_prediction = None
            training_state.num_iter_without_reset += 1

        return training_states
