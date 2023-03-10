import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from omegaconf import DictConfig

from nuplan.planning.training.modeling.closed_loop_utilities.abstract_training_controller import \
    AbstractTrainingController
from nuplan.planning.training.modeling.metrics.planning_metrics import \
    AbstractTrainingMetric
from nuplan.planning.training.modeling.objectives.abstract_objective import \
    aggregate_objectives
from nuplan.planning.training.modeling.objectives.imitation_objective import \
    AbstractObjective
from nuplan.planning.training.modeling.torch_module_wrapper import \
    TorchModuleWrapper
from nuplan.planning.training.modeling.types import (FeaturesType,
                                                     ScenarioListType,
                                                     TargetsType)

from nuplan_extent.planning.training.modeling.sequential_utilities.feature_cache import FeatureCacheContainer
from .lightning_module_wrapper import LightningModuleWrapper

logger = logging.getLogger(__name__)


class LightningModuleWrapperCloseloop(LightningModuleWrapper):
    """
    Lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        objectives: List[AbstractObjective],
        metrics: List[AbstractTrainingMetric],
        batch_size: int,
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        warm_up_lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = 'mean',
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__(
            model=model,
            objectives=objectives,
            metrics=metrics,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            warm_up_lr_scheduler=warm_up_lr_scheduler,
            objective_aggregate_mode=objective_aggregate_mode,
        )

        self.batch_size = batch_size

        # closed loop essentials
        self._token2state: Dict[str, AbstractTrainingController] = {}  # State memory for every scenario

        # sequential model essentials
        self._token2cache: Dict[str, FeatureCacheContainer] = {}

    def _step(self, batch: Tuple[FeaturesType, TargetsType], prefix: str, batch_idx: int) -> Dict[str, Any]:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch

        predictions = self.forward(features)
        objectives = self._compute_objectives(predictions, targets, scenarios)
        # for k, v in objectives.items():
        #     print(k, v)
        metrics = self._compute_metrics(predictions, targets)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        self._log_step(loss, objectives, metrics, prefix, batch_idx=batch_idx)

        return_dict = {
            "loss": loss,
            "trajectory": predictions["trajectory"],
        }

        if "out_feature" in predictions:
            return_dict["out_feature"] = predictions["out_feature"].detach()
        if 'bev_feature' in predictions:
            return_dict["bev_feature"] = predictions["bev_feature"].to_device("cpu").detach()

        if hasattr(self, "closeloop_data"):
            return_dict.update(self.closeloop_data["logged_objects"])
        return return_dict

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = 'loss',
        batch_idx: int = 0,
        **kwargs
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log('idx', batch_idx, prog_bar=True)
        self.log(f'loss/{prefix}_{loss_name}', loss)

        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value)

        for key, value in metrics.items():
            self.log(f'metrics/{prefix}_{key}', value)

        for key, value in kwargs.items():
            self.log(f'{key}', value)

    def training_step(self, batch: Tuple[FeaturesType, TargetsType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'train', batch_idx)

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'val', batch_idx)

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'test', batch_idx)

    def on_epoch_start(self) -> None:
        # Ensures all CUDA tensors are recycled
        if self._token2state is not None:
            logger.info("Resetting _token2state before epoch starts.")
            self._token2state.clear()
