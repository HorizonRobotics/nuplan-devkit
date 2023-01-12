from typing import Dict, List

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractModelFeature, AbstractScenario)


class MissingFeature(Exception):
    """
    Exception used when a features is not present
    """

    pass


FeaturesType = Dict[str, AbstractModelFeature]
TargetsType = Dict[str, AbstractModelFeature]
ScenarioListType = List[AbstractScenario]
EgoStateListType = List[EgoState]
TensorFeaturesType = Dict[str, torch.Tensor]


def move_features_type_to_device(batch: FeaturesType, device: torch.device) -> FeaturesType:
    """
    Move all features to a device
    :param batch: batch of features
    :param device: new device
    :return: batch moved to new device
    """
    output = {}
    for key, value in batch.items():
        if isinstance(value, (int, float)):
            output[key] = value
        elif isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value.to_device(device)
    return output
