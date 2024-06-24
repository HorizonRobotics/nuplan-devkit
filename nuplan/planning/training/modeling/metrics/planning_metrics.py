from typing import List

import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class AverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        if predicted_trajectory.xy.shape[1] == targets_trajectory.xy.shape[1]:
            return torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1).mean()
        else:
            # Get the minimum number of points
            min_points = min(predicted_trajectory.xy.shape[1], targets_trajectory.xy.shape[1])

            # Slice the trajectories to the minimum number of points
            predicted_slice = predicted_trajectory.xy[:, :min_points, :]
            targets_slice = targets_trajectory.xy[:, :min_points, :]

            # Compute the Euclidean distance and return the mean
            return torch.norm(predicted_slice - targets_slice, dim=-1).mean()


class FinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        if predicted_trajectory.xy.shape[1] == targets_trajectory.xy.shape[1]:
            return torch.norm(predicted_trajectory.terminal_position - targets_trajectory.terminal_position, dim=-1).mean()
        else:
            min_points = min(predicted_trajectory.xy.shape[1], targets_trajectory.xy.shape[1])
            predicted_slice = predicted_trajectory.xy[:, min_points-1, :]
            targets_slice = targets_trajectory.xy[:, min_points-1, :]
            return torch.norm(predicted_slice - targets_slice, dim=-1).mean()


class AverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        if predicted_trajectory.heading.shape[1] == targets_trajectory.heading.shape[1]:
            errors = torch.abs(predicted_trajectory.heading - targets_trajectory.heading)
            return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()
        else:
            # Get the minimum number of points
            min_points = min(predicted_trajectory.heading.shape[1], targets_trajectory.heading.shape[1])

            # Slice the trajectories to the minimum number of points
            predicted_slice = predicted_trajectory.heading[:, :min_points]
            targets_slice = targets_trajectory.heading[:, :min_points]

            # Compute the heading error and return the mean
            errors = torch.abs(predicted_slice - targets_slice)
            return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()


class FinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        if predicted_trajectory.heading.shape[1] == targets_trajectory.heading.shape[1]:
            errors = torch.abs(predicted_trajectory.terminal_heading - targets_trajectory.terminal_heading)
            return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()
        else:
            min_points = min(predicted_trajectory.heading.shape[1], targets_trajectory.heading.shape[1])
            predicted_slice = predicted_trajectory.heading[:, min_points-1]
            targets_slice = targets_trajectory.heading[:, min_points-1]
            errors = torch.abs(predicted_slice - targets_slice)
            return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()

            