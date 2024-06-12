import logging
from typing import List, Optional, Tuple

import torch.utils.data

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)

def decorate_for_sparse4d(dataset_cls):
    original_init = dataset_cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.flag = self._feature_preprocessor.feature_builders[0].flag
        self.keep_consistent_seq_aug = self._feature_preprocessor.feature_builders[0].keep_consistent_seq_aug
        self.get_augmentation = self._feature_preprocessor.feature_builders[0].get_augmentation

    def new__getitem__(self, idx):
        # Copied from nuscenes_3d_dataset
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        scenario = self._scenarios[idx]

        features, targets, _ = self._feature_preprocessor.compute_features((scenario, aug_config))

        if self._augmentors is not None:
            for augmentor in self._augmentors:
                augmentor.validate(features, targets)
                features, targets = augmentor.augment(features, targets, scenario)

        features = {key: value.to_feature_tensor() for key, value in features.items()}
        targets = {key: value.to_feature_tensor() for key, value in targets.items()}
        scenarios = [scenario]

        return features, targets, scenarios        
    
    dataset_cls.__init__ = new_init
    dataset_cls.__getitem__ = new__getitem__

    return dataset_cls

@decorate_for_sparse4d
class ScenarioDataset(torch.utils.data.Dataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(
        self,
        scenarios: List[AbstractScenario],
        feature_preprocessor: FeaturePreprocessor,
        augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        """
        Initializes the scenario dataset.
        :param scenarios: List of scenarios to use as dataset examples.
        :param feature_preprocessor: Feature and targets builder that converts samples to model features.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        if len(scenarios) == 0:
            logger.warning('The dataset has no samples')

        self._scenarios = scenarios
        self._feature_preprocessor = feature_preprocessor
        self._augmentors = augmentors

    def __getitem__(self, idx: int) -> Tuple[FeaturesType, TargetsType, ScenarioListType]:
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: model features and targets
        """
        scenario = self._scenarios[idx]

        features, targets, _ = self._feature_preprocessor.compute_features(scenario)

        if self._augmentors is not None:
            for augmentor in self._augmentors:
                augmentor.validate(features, targets)
                features, targets = augmentor.augment(features, targets, scenario)

        features = {key: value.to_feature_tensor() for key, value in features.items()}
        targets = {key: value.to_feature_tensor() for key, value in targets.items()}
        scenarios = [scenario]

        return features, targets, scenarios

    def __len__(self) -> int:
        """
        Returns the size of the dataset (number of samples)

        :return: size of dataset
        """
        return len(self._scenarios)
