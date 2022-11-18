import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler
from torch.utils.data.dataloader import default_collate

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.script.builders.lr_scheduler_builder import \
    build_lr_scheduler
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.training.modeling.closed_loop_utilities.abstract_training_controller import \
    AbstractTrainingController
from nuplan.planning.training.modeling.closed_loop_utilities.raster_repainter import \
    repaint_raster
from nuplan.planning.training.modeling.closed_loop_utilities.target_trajectory_recomputer import \
    TargetTrajectoryRecomputer
from nuplan.planning.training.modeling.closed_loop_utilities.utils import \
    get_relative_pose_matrices_to
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
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory

logger = logging.getLogger(__name__)

def _get_batched_pose_matrices(ego_states: List[EgoState]) -> torch.Tensor:
    """Convert list of ego states to a tensor of pose matrices.

    The converted tensor will have a shape of Nx3x3.

    :param ego_states: A list of EgoState objects.
    :return: A tensor representing batched matrices.
    """
    matrices = np.stack([i.rear_axle.as_matrix() for i in ego_states])
    return torch.from_numpy(matrices)

class LightningModuleWrapperV2(pl.LightningModule):
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
        ego_controller: Optional[AbstractTrainingController] = None,
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
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode
        self.batch_size = batch_size

        self._ego_controller = ego_controller
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time

        self._feature_builders = model.get_list_of_required_feature()

        # closed loop essentials
        self.is_closed_loop_model = False
        self._trajectory_smoother = None
        self._last_step_prediction = None # Last step's prediction
        self._token2state: Dict[str, AbstractTrainingController] = None # State memory for every scenario
        self._setup_closed_loop_if_necessary()

        # Validate metrics objectives and model
        model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
        for objective in self.objectives:
            for feature in objective.get_list_of_required_target_types():
                assert feature in model_targets, f"Objective target: \"{feature}\" is not in model computed targets!"
        for metric in self.metrics:
            for feature in metric.get_list_of_required_target_types():
                assert feature in model_targets, f"Metric target: \"{feature}\" is not in model computed targets!"

    def _setup_closed_loop_if_necessary(self):
        """Set up closed loop training"""
        if self._ego_controller is not None:
            self.is_closed_loop_model = True
            self._trajectory_smoother = TargetTrajectoryRecomputer(
                int(self._future_horizon / self._step_interval),
                self._step_interval
            )
            self._token2state = {}

    def _step(self, batch: Tuple[FeaturesType, TargetsType], prefix: str, batch_idx: int) -> Dict[str, Any]:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        additional_logged_objects = {}
        # Closed-loop extra computations
        if self.is_closed_loop_model and prefix == 'train':
            current_iterations = features["current_iteration"] # int tensor
            gt_ego_states = [i.get_ego_state_at_iteration(j.item()) for i, j in zip(scenarios, current_iterations)]
            scenario_tokens = [i.token for i in scenarios]
            N = len(scenario_tokens)
            
            # Add unseen scenario, or update a seen scenario
            for token, iteration, ego_state in zip(scenario_tokens, current_iterations, gt_ego_states):
                if token not in self._token2state:
                    # Unseen, input data and controller are at the same time.
                    new_controller = copy.deepcopy(self._ego_controller)
                    new_controller.initialize(state=ego_state, current_iteration=iteration)
                    self._token2state[token] = new_controller
                else:
                    # Previously seen, input data must be 1 step ahead of controller.
                    iter_diff = iteration - self._token2state[token].current_iteration
                    if iter_diff != 1:
                        logger.warning(f"{self.local_rank}, {token} input iter is "
                        f"{iteration.item()}, recorded iter is "
                        f"{self._token2state[token].current_iteration.item()}")
                    self._token2state[token].update(ego_state.time_point)

            # At this point, time must be aligned.

            # Gather absoluate states (both cl and gt)
            cl_pose_matrices = _get_batched_pose_matrices(
                [self._token2state[i].state for i in scenario_tokens]
            ).to(self.device)
            gt_pose_matrices = _get_batched_pose_matrices(gt_ego_states).to(self.device)

            # Compute relative poses
            pose_matrices = get_relative_pose_matrices_to(cl_pose_matrices, gt_pose_matrices)
            additional_logged_objects['closed_loop_states'] = cl_pose_matrices.clone()
            additional_logged_objects['gt_states'] = gt_pose_matrices.clone()
            additional_logged_objects['relative_poses'] = pose_matrices.clone()
            new_raster, dist_bound, rot_bound, dist, rot= repaint_raster(features["raster"].data, pose_matrices)
            # recompute trajectory
            updated_target, reset_sample_index, num_corrections = [], [], 0
            for i, scenario_token in enumerate(scenario_tokens):
                ref_traj = targets["trajectory"].data[i]
                curr_iter = current_iterations[i]
                if dist_bound[i] and rot_bound[i]:
                    speed = self._token2state[scenario_token].state.dynamic_car_state.speed
                    solve_success, new_traj = self._trajectory_smoother.recompute(
                        ref_traj,
                        pose_matrices[i, ...],
                        speed
                    )
                    if solve_success:
                        updated_target.append(new_traj)
                    else:
                        self._token2state[scenario_token].set(gt_ego_states[i], curr_iter)
                        new_raster[i] = features["raster"].data[i]
                        updated_target.append(ref_traj)
                        num_corrections += 1
                        reset_sample_index.append(i)
                else:
                    reason = f"Sample {i} in batch {batch_idx} reset at iter {current_iterations[i].item()}."
                    if not dist_bound[i]: 
                        reason = reason + f" Dist. off by {dist[i].item():.4f}m."
                    if not rot_bound[i]:
                        reason = reason + f" Rot. off by {rot[i].item() * 180 / math.pi:.4f} deg."
                    reason = reason + f" Recorded iter: {self._token2state[scenario_token].current_iteration.item()}."
                    reason = reason + f" Last reset was {self._token2state[scenario_token].num_iter_without_reset} iter ago."
                    logger.info(reason)
                    self._token2state[scenario_token].set(gt_ego_states[i], curr_iter)
                    updated_target.append(ref_traj)
                    num_corrections += 1
                    reset_sample_index.append(i)
            
            features["raster"].data = new_raster
            targets["trajectory"].data = default_collate(updated_target)
            additional_logged_objects["batch_corrections"] = num_corrections
            additional_logged_objects["reset_sample_index"] = reset_sample_index

        predictions = self.forward(features)
        objectives = self._compute_objectives(predictions, targets, scenarios)
        metrics = self._compute_metrics(predictions, targets)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        # Store predicted trajectory.
        # Cannot update here, because we don't know what the next time point is.
        if self.is_closed_loop_model and prefix == 'train':
            trajectory_predicted = cast(Trajectory, predictions["trajectory"])
            trajectory_per_scene = [
                i[0].cpu().numpy() for i in torch.tensor_split(
                    trajectory_predicted.data.detach(),
                    N,
                    dim=0
                )
            ]
            for i in range(N):
                scenario_token = scenario_tokens[i]
                scenario_trajectory = trajectory_per_scene[i]
                self._token2state[scenario_token].last_trajectory = scenario_trajectory

        self._log_step(loss, objectives, metrics, prefix, batch_idx=batch_idx)

        return_dict = {
            "loss": loss, 
            "trajectory": predictions["trajectory"],
        }
        return_dict.update(additional_logged_objects)
        return return_dict

    def _compute_objectives(
        self, predictions: TargetsType, targets: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: model's output signal
        :param targets: supervisory signal
        :return: dictionary of objective names and values
        """
        return {objective.name(): objective.compute(predictions, targets, scenarios) for objective in self.objectives}

    def _compute_metrics(self, predictions: TargetsType, targets: TargetsType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        return {metric.name(): metric.compute(predictions, targets) for metric in self.metrics}

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

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError("To train, optimizer must not be None.")

        # Get optimizer
        optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        # Log the optimizer used
        logger.info(f'Using optimizer: {self.optimizer._target_}')

        # Get lr_scheduler
        lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )

        optimizer_dict: Dict[str, Any] = {}
        optimizer_dict['optimizer'] = optimizer
        if lr_scheduler_params:
            logger.info(f'Using lr_schedulers {lr_scheduler_params}')
            optimizer_dict['lr_scheduler'] = lr_scheduler_params

        return optimizer_dict if 'lr_scheduler' in optimizer_dict else optimizer_dict['optimizer']

    def on_epoch_start(self) -> None:
        # Ensures all CUDA tensors are recycled
        if self._token2state is not None:
            logger.info("Resetting _token2state before epoch starts.")
            self._token2state.clear()
        