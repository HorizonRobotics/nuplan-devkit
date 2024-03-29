import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.data.sampler import WeightedRandomSampler

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_loader.closed_loop_scenario_dataset import ClosedLoopScenarioDatasetV2, ClosedLoopScenarioDatasetV3
from nuplan.planning.training.data_loader.distributed_sampler_wrapper import DistributedSamplerWrapper
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.modeling.types import FeaturesType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

DataModuleNotSetupError = RuntimeError('Data module has not been setup, call "setup()"')


def create_dataset(
    samples: List[AbstractScenario],
    feature_preprocessor: FeaturePreprocessor,
    dataset_fraction: float,
    dataset_name: str,
    augmentors: Optional[List[AbstractAugmentor]] = None,
    closed_loop_batch_size: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """
    Create a dataset from a list of samples.
    :param samples: List of dataset candidate samples.
    :param feature_preprocessor: Feature preprocessor object.
    :param dataset_fraction: Fraction of the dataset to load.
    :param dataset_name: Set name (train/val/test).
    :param scenario_type_loss_weights: Dictionary of scenario type loss weights.
    :param augmentors: List of augmentor objects for providing data augmentation to data samples.
    :param closed_loop_batch_size: per-gpu batch size used during closed-loop training.
    :return: The instantiated torch dataset.
    """
    # Sample the desired fraction from the total samples
    num_keep = int(len(samples) * dataset_fraction)
    selected_scenarios = random.sample(samples, num_keep)
    
    # manually duplicate scenarios if there is only one scenario(for DDP)
    # if len(selected_scenarios) == 1:
    #     import copy
    #     for i in range(15):
    #         s = copy.deepcopy(selected_scenarios[0])
    #         selected_scenarios.append(s)

    if closed_loop_batch_size is not None:
        logger.info(f"Creating closed-loop dataset...")
        return ClosedLoopScenarioDatasetV2(
            batch_size=closed_loop_batch_size,
            scenarios=selected_scenarios,
            feature_preprocessor=feature_preprocessor,
            augmentors=augmentors,
        )
    else:
        logger.info(f"Creating open-loop dataset. Number of samples in {dataset_name} set: {len(selected_scenarios)}")
        return ScenarioDataset(
            scenarios=selected_scenarios,
            feature_preprocessor=feature_preprocessor,
            augmentors=augmentors,
        )


def distributed_weighted_sampler_init(
    scenario_dataset: ScenarioDataset, scenario_sampling_weights: Dict[str, float], replacement: bool = True
) -> WeightedRandomSampler:
    """
    Initiliazes WeightedSampler object with sampling weights for each scenario_type and returns it.
    :param scenario_dataset: ScenarioDataset object
    :param replacement: Samples with replacement if True. By default set to True.
    return: Initialized Weighted sampler
    """
    scenarios = scenario_dataset._scenarios
    if not replacement:  # If we don't sample with replacement, then all sample weights must be nonzero
        assert all(
            w > 0 for w in scenario_sampling_weights.values()
        ), "All scenario sampling weights must be positive when sampling without replacement."

    default_scenario_sampling_weight = 1.0

    scenario_sampling_weights_per_idx = [
        scenario_sampling_weights[scenario.scenario_type]
        if scenario.scenario_type in scenario_sampling_weights
        else default_scenario_sampling_weight
        for scenario in scenarios
    ]

    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=scenario_sampling_weights_per_idx,
        num_samples=len(scenarios),
        replacement=replacement,
    )

    distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)
    return distributed_weighted_sampler


class DataModule(pl.LightningDataModule):
    """
    Datamodule wrapping all preparation and dataset creation functionality.
    """

    def __init__(
        self,
        feature_preprocessor: FeaturePreprocessor,
        splitter: AbstractSplitter,
        all_scenarios: List[AbstractScenario],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        dataloader_params: Dict[str, Any],
        scenario_type_sampling_weights: DictConfig,
        worker: WorkerPool,
        augmentors: Optional[List[AbstractAugmentor]] = None,
        val_augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        """
        Initialize the class.
        :param feature_preprocessor: Feature preprocessor object.
        :param splitter: Splitter object used to retrieve lists of samples to construct train/val/test sets.
        :param train_fraction: Fraction of training examples to load.
        :param val_fraction: Fraction of validation examples to load.
        :param test_fraction: Fraction of test examples to load.
        :param dataloader_params: Parameter dictionary passed to the dataloaders.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        assert train_fraction > 0.0, "Train fraction has to be larger than 0!"
        assert val_fraction > 0.0, "Validation fraction has to be larger than 0!"
        assert test_fraction >= 0.0, "Test fraction has to be larger/equal than 0!"

        # Datasets
        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None

        # Feature computation
        self._feature_preprocessor = feature_preprocessor

        # Data splitter train/test/val
        self._splitter = splitter

        # Fractions
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # Data loader for train/val/test
        self._dataloader_params = dataloader_params

        # Extract all samples
        self._all_samples = all_scenarios
        assert len(self._all_samples) > 0, 'No samples were passed to the datamodule'

        # Scenario sampling weights
        self._scenario_type_sampling_weights = scenario_type_sampling_weights

        # Augmentation setup
        self._augmentors = augmentors
        self._val_augmentors = val_augmentors

        # Worker for multiprocessing to speed up initialization of datasets
        self._worker = worker

        # sequential dataset
        self._sequential_train = self._dataloader_params.get('sequential_train', False)
        self._sequential_val = self._dataloader_params.get('sequential_val', False)
        self._sequential_test = self._dataloader_params.get('sequential_test', False)
        if 'sequential_train' in self._dataloader_params:
            del self._dataloader_params.sequential_train
        if 'sequential_val' in self._dataloader_params:
            del self._dataloader_params.sequential_val
        if 'sequential_test' in self._dataloader_params:
            del self._dataloader_params.sequential_test

    @property
    def feature_and_targets_builder(self) -> FeaturePreprocessor:
        """Get feature and target builders."""
        return self._feature_preprocessor

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset for each target set depending on the training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        if stage is None:
            return

        if stage == 'fit':
            # Training Dataset
            train_samples = self._splitter.get_train_samples(self._all_samples, self._worker)
            assert len(train_samples) > 0, 'Splitter returned no training samples'

            self._train_set = create_dataset(
                train_samples,
                self._feature_preprocessor,
                self._train_fraction,
                "train",
                self._augmentors,
                self._dataloader_params.batch_size if self._sequential_train else None
            )

            # Validation Dataset
            val_samples = self._splitter.get_val_samples(self._all_samples, self._worker)
            assert len(val_samples) > 0, 'Splitter returned no validation samples'

            val_batch_size = self._dataloader_params.batch_size if self._sequential_val else None
            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
                self._val_augmentors,
                val_batch_size,
            )
        elif stage == 'test':
            # Testing Dataset
            test_samples = self._splitter.get_test_samples(self._all_samples, self._worker)
            assert len(test_samples) > 0, 'Splitter returned no test samples'

            test_batch_size = self._dataloader_params.batch_size if self._sequential_test else None
            self._test_set = create_dataset(
                test_samples,
                self._feature_preprocessor,
                self._test_fraction,
                "test",
                None,
                test_batch_size,
            )
        else:
            raise ValueError(f'Stage must be one of ["fit", "test"], got ${stage}.')

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after a training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the training dataloader.
        :raises RuntimeError: If this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._train_set is None:
            raise DataModuleNotSetupError

        # Initialize weighted sampler
        if self._scenario_type_sampling_weights.enable:
            weighted_sampler = distributed_weighted_sampler_init(
                scenario_dataset=self._train_set,
                scenario_sampling_weights=self._scenario_type_sampling_weights.scenario_type_weights,
            )
        else:
            weighted_sampler = None

        if self._sequential_train:
            return torch.utils.data.DataLoader(
                dataset=self._train_set,
                **self._dataloader_params,
                sampler=torch.utils.data.distributed.DistributedSampler(
                    self._train_set,
                    shuffle=False,
                ),
                collate_fn=FeatureCollate(),
            )
        else:
            return torch.utils.data.DataLoader(
                dataset=self._train_set,
                shuffle=weighted_sampler is None, 
                collate_fn=FeatureCollate(),
                sampler=weighted_sampler,
                **self._dataloader_params,
            )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the validation dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._val_set is None:
            raise DataModuleNotSetupError

        if self._sequential_val:
            return torch.utils.data.DataLoader(
                dataset=self._val_set,
                **self._dataloader_params,
                sampler=torch.utils.data.distributed.DistributedSampler(
                    self._val_set,
                    shuffle=False,
                ),
                collate_fn=FeatureCollate(),
            )
        else:
            return torch.utils.data.DataLoader(
                dataset=self._val_set,
                collate_fn=FeatureCollate(),
                **self._dataloader_params,
            )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the test dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._test_set is None:
            raise DataModuleNotSetupError

        if self._sequential_test:
            return torch.utils.data.DataLoader(
                dataset=self._test_set,
                **self._dataloader_params,
                sampler=torch.utils.data.distributed.DistributedSampler(
                    self._test_set,
                    shuffle=False,
                ),
                collate_fn=FeatureCollate()
            )
        else:
            return torch.utils.data.DataLoader(
                dataset=self._test_set,
                collate_fn=FeatureCollate(),
                **self._dataloader_params,
            )

    def transfer_batch_to_device(
        self, batch: Tuple[FeaturesType, ...], device: torch.device
    ) -> Tuple[FeaturesType, ...]:
        """
        Transfer a batch to device.
        :param batch: Batch on origin device.
        :param device: Desired device.
        :return: Batch in new device.
        """
        return tuple(
            (move_features_type_to_device(batch[0], device), move_features_type_to_device(batch[1], device), batch[2])
        )
