import logging
import random
from typing import List, Optional, Tuple

import torch.utils.data
import torch.distributed as dist

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import \
    AbstractAugmentor
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.feature_preprocessor import \
    FeaturePreprocessor

logger = logging.getLogger(__name__)

class ClosedLoopScenarioDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size: int,
        scenarios: List[AbstractScenario],
        feature_preprocessor: FeaturePreprocessor,
        augmentors: Optional[List[AbstractAugmentor]] = None,
        **kwargs
    ) -> None:
        """
        """
        super().__init__()

        if len(scenarios) == 0:
            logger.warning('The dataset has no samples.')

        assert all([i.get_number_of_iterations() > 1 for i in scenarios]), \
            "Not all scenarios contain multiple timesteps."

        self._feature_preprocessor = feature_preprocessor
        self._augmentors = augmentors

        effective_batch = dist.get_world_size() * batch_size
        if len(scenarios) % effective_batch != 0:
            scenarios_to_drop = len(scenarios) - len(scenarios) // effective_batch * effective_batch
            logger.warning(f"Number of scenarios {len(scenarios)} is not "
            f"divisible by world_size x batch_size. Will drop the last {scenarios_to_drop} scenarios.")
            scenarios = scenarios[:-scenarios_to_drop]
        self._idx2token = {}
        self._token2scenario = {i.token: i for i in scenarios}
        # nuplan_mark
        self._scenario_max_len = min([i.get_number_of_iterations() for i in scenarios]) - scenarios[0].total_steps
        self._length = self._scenario_max_len * len(scenarios)
        self._batch_size = batch_size

        logger.info("Indexing scenarios...")
        token_sequence = [i.token for i in scenarios]
        random.shuffle(token_sequence)
        for time in range(self._scenario_max_len):
            for i, sid in enumerate(token_sequence):
                scenario = self._token2scenario[sid]
                idx = time * len(scenarios) + i
                self._idx2token[idx] = '_'.join([scenario.token, str(time)])
        logger.info("Indexing scenarios...DONE!")
        logger.info(f"Scenario count: {len(scenarios)}")
        logger.info(f"Scenario length: {self._scenario_max_len}"),
        logger.info(f"Dataset size: {self._length}")
        logger.info(f"Batch size: {self._batch_size}")
        logger.info(f"Total batch count: {self._length // self._batch_size + 1}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int) -> Tuple[FeaturesType, TargetsType, ScenarioListType]:
        """
        """
        token_time = self._idx2token[idx]  
        split_pos = token_time.rfind('_')
        token, iteration = token_time[:split_pos], token_time[split_pos+1:]
        iteration = int(iteration)
        scenario = self._token2scenario[token]

        features, targets, _ = self._feature_preprocessor.compute_features(scenario, iteration)

        # TODO: enable augmentors
        # if self._augmentors is not None:
        #     for augmentor in self._augmentors:
        #         augmentor.validate(features, targets)
        #         features, targets = augmentor.augment(features, targets, scenario.scenario)

        features = {key: value.to_feature_tensor() for key, value in features.items()}
        features["current_iteration"] = iteration
        targets = {key: value.to_feature_tensor() for key, value in targets.items()}
        scenarios = [scenario]

        return features, targets, scenarios