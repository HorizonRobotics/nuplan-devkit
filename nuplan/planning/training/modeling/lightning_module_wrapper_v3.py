import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import pdb

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from nuplan_extent.planning.training.closed_loop.batched_nonlinear_smoother import \
    BatchedNonLinearSmoother
from nuplan_extent.planning.training.closed_loop.controllers.abstract_training_controller import \
    AbstractTrainingController
from nuplan_extent.planning.training.closed_loop.raster_repainter import \
    repaint_raster
from nuplan_extent.planning.training.closed_loop.target_trajectory_recomputer import \
    TargetTrajectoryRecomputer
from nuplan_extent.planning.training.closed_loop.utils.torch_util import (
    TrainingState, get_heading, get_relative_pose_matrices_to,
    replace_out_of_bound_matrices_with_identity)
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.script.builders.lr_scheduler_builder import \
    build_lr_scheduler
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

class LightningModuleWrapperV3(pl.LightningModule):
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
        self.save_hyperparameters(ignore=["model", "ego_controller"])

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
        self._updated_flag = False

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
            self._ego_controller.initialize(device=self.device)
            self.is_closed_loop_model = True
            # self._trajectory_smoother = TargetTrajectoryRecomputer(
            #     int(self._future_horizon / self._step_interval),
            #     self._step_interval
            # )
            self._trajectory_smoother = BatchedNonLinearSmoother(
                trajectory_len=int(self._future_horizon / self._step_interval),
                dt=self._step_interval,
                batch_size=self.batch_size,
                device=self.device
            )
            self._token2state: Dict[str, TrainingState] = {}
            self._updated_flag = False
    
    def on_train_start(self) -> None:
        self._setup_closed_loop_if_necessary()

    def update_controllers(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType]) -> None:
        """Update closed loop controllers"""
        features, _, scenarios = batch
        current_iterations = features["current_iteration"] # int tensor
        scenario_tokens = [i.token for i in scenarios]
        gt_ego_states = features["ego_state"]
        # Add unseen scenario, or update a seen scenario
        to_be_updated_tokens, incoming_gt_ego_states = [], []
        for token, iteration, ego_state in zip(scenario_tokens, current_iterations, gt_ego_states):
            if token not in self._token2state:
                # Unseen, input data and controller are at the same time.
                training_state = TrainingState(
                    token=token,
                    state=deque([ego_state], maxlen=10), 
                    last_prediction=None, 
                    current_iteration=iteration.item()
                )
                self._token2state[token] = training_state
            else:
                # Previously seen, input data must be 1 step ahead of controller.
                iter_diff = iteration - self._token2state[token].current_iteration
                if iter_diff != 1:
                    logger.warning(f"{self.local_rank}, {token} input iter is "
                    f"{iteration.item()}, recorded iter is "
                    f"{self._token2state[token].current_iteration}")
                to_be_updated_tokens.append(token)
                incoming_gt_ego_states.append(ego_state)
        to_be_updated_states = [self._token2state[i] for i in to_be_updated_tokens]
        assert len(to_be_updated_states) == 0 or all([i.last_prediction is not None for i in to_be_updated_states])
        if len(to_be_updated_states) > 0:
            updated_states = self._ego_controller.update(to_be_updated_states, incoming_gt_ego_states)
            for updated_state in updated_states:
                self._token2state[updated_state.token] = updated_state

        self._updated_flag = True


    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str, batch_idx: int) -> Dict[str, Any]:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        additional_logged_objects = {}
        pose_matrices, gt_ego_states, is_in_bound = None, None, None
        # Closed-loop extra computations
        if self.is_closed_loop_model and prefix == 'train':
            current_iterations = features["current_iteration"] # int tensor
            gt_ego_states = features["ego_state"]
            scenario_tokens = [i.token for i in scenarios]
            N = len(scenario_tokens)

            if not self._updated_flag:
                self.update_controllers(batch)
 
            # Sanity check
            for token, iteration in zip(scenario_tokens, current_iterations):
                iter_diff = iteration - self._token2state[token].current_iteration
                if iter_diff != 0:
                    logger.warning(f"rank {self.local_rank}, {token} input iter is "
                    f"{iteration.item()}, recorded iter is "
                    f"{self._token2state[token].current_iteration}, expected diffrence must be 0.")
                    raise RuntimeError("Input sequence is not in correct order.")

            # ====== At this point, time must be aligned ======

            # Gather absoluate states (both cl and gt)
            cl_pose_matrices = _get_batched_pose_matrices(
                [self._token2state[i].state[-1] for i in scenario_tokens]
            ).to(self.device).to(self.dtype)
            gt_pose_matrices = _get_batched_pose_matrices(gt_ego_states).to(self.device).to(self.dtype)

            # Compute relative poses
            pose_matrices = get_relative_pose_matrices_to(cl_pose_matrices, gt_pose_matrices)
            pose_matrices, is_in_bound = replace_out_of_bound_matrices_with_identity(pose_matrices)
 
            additional_logged_objects['closed_loop_states'] = cl_pose_matrices.clone()
            additional_logged_objects['gt_states'] = gt_pose_matrices.clone()
            additional_logged_objects['relative_poses'] = pose_matrices.clone()
            new_raster = repaint_raster(features["raster"].data, pose_matrices)

            # recompute trajectory
            reset_sample_index, num_corrections = [], 0
            x = pose_matrices[:, 0, 2]
            y = pose_matrices[:, 1, 2]
            h = get_heading(pose_matrices[:, 1, 0], pose_matrices[:, 0, 0])
            v = torch.tensor([self._token2state[i].state[-1].dynamic_car_state.speed for i in scenario_tokens], device=self.device)
            x_curr = torch.stack([x, y, h, v], dim=-1)
            batch_ref_traj = torch.nn.functional.pad(
                targets["trajectory"].data,
                pad=(0,0,1,0),
                mode='constant',
                value=0.
            )
            updated_target = self._trajectory_smoother.solve(x_curr, batch_ref_traj)
            
            features["raster"].data = new_raster
            targets["trajectory"].data = updated_target
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
                i[0] for i in torch.tensor_split(
                    trajectory_predicted.data.detach(),
                    N,
                    dim=0
                )
            ]
            for i in range(N):
                scenario_token = scenario_tokens[i]
                scenario_trajectory = trajectory_per_scene[i]
                self._token2state[scenario_token].last_prediction = scenario_trajectory
                # reset training state if out of bound
                if not is_in_bound[i]:
                    self._token2state[scenario_token].state = deque([gt_ego_states[i]], maxlen=10)
                    self._token2state[scenario_token].forced_reset_count += 1
                    self._token2state[scenario_token].num_iter_without_reset = 0

        self._log_step(loss, objectives, metrics, prefix, batch_idx=batch_idx)

        return_dict = {
            "loss": loss, 
            "trajectory": predictions["multimode_trajectory_6"],
        }
        return_dict.update(additional_logged_objects)
        self._updated_flag = False
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
        self.log('idx', batch_idx, prog_bar=True, batch_size=self.batch_size)
        self.log(f'loss/{prefix}_{loss_name}', loss, batch_size=self.batch_size)

        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value, batch_size=self.batch_size)

        for key, value in metrics.items():
            self.log(f'metrics/{prefix}_{key}', value, batch_size=self.batch_size)

        for key, value in kwargs.items():
            self.log(f'{key}', value, batch_size=self.batch_size)

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

    def on_train_epoch_start(self) -> None:
        # Ensures all CUDA tensors are recycled
        if self._token2state is not None:
            logger.info("Resetting _token2state before epoch starts.")
            self._token2state.clear()
        