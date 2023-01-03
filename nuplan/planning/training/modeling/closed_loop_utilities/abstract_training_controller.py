import abc

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint


class AbstractTrainingController(object):
    """
    Abstract class for controllers during training.
    """

    def __init__(self, step_time: float, future_horizon: float):
        self._step_interval = step_time
        self._future_horizon = future_horizon
        self.state = None
        self.current_iteration = None
        self.last_trajectory = None

    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the controller.

        :param state: EgoState object.
        :param current_iteration: a single-valued tensor representing which
            iteration it is.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(self, **kwargs) -> None:
        """Set the current status of the controller.

        Args:
            state (EgoState): current ego-vehicle's state.
            current_iteration (torch.Tensor): current iteration.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, **kwargs) -> None:
        """Update controller's state to the next time point.

        :param timepoint: the next time point.
        """
        raise NotImplementedError()
