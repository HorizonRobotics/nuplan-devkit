from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class ImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        if isinstance(predictions["trajectory"], list):
            predicted_trajectories = [cast(Trajectory, i) for i in predictions["trajectory"]]
        else:
            predicted_trajectories = [cast(Trajectory, predictions["trajectory"])]
        targets_trajectory = cast(Trajectory, targets["trajectory"])
        loss_weights = extract_scenario_type_weight(
            scenarios, self._scenario_type_loss_weighting, device=predicted_trajectories[-1].xy.device
        )

        losses = []
        for predicted_trajectory in predicted_trajectories:
            broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.xy.dim() - 1)])
            broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
            broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.heading.dim() - 1)])
            broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)

            weighted_xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * broadcasted_loss_weights_xy
            weighted_heading_loss = (
            self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading)
                * broadcasted_loss_weights_heading
            )
            # Assert that broadcasting was done correctly
            assert weighted_xy_loss.size() == predicted_trajectory.xy.size()
            assert weighted_heading_loss.size() == predicted_trajectory.heading.size()

            losses.append(self._weight * (torch.mean(weighted_xy_loss) + torch.mean(weighted_heading_loss)))

        loss = sum(losses) / len(losses)
        return loss
