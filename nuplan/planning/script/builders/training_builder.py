import logging
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from ray_lightning import RayStrategy
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies.strategy import Strategy

from nuplan.planning.script.builders.data_augmentation_builder import (
    build_agent_augmentor,
)
from nuplan.planning.script.builders.objectives_builder import build_objectives
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.training_callback_builder import build_callbacks
from nuplan.planning.script.builders.training_metrics_builder import build_training_metrics
from nuplan.planning.script.builders.aggregated_metrics_builder import build_aggregated_metrics
from nuplan.planning.script.builders.utils.utils_checkpoint import extract_last_checkpoint_from_experiment
from nuplan.planning.training.data_loader.datamodule import DataModule
from nuplan.planning.training.modeling.lightning_module_wrapper_closeloop import (
    LightningModuleWrapperCloseloop,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

def build_lightning_datamodule(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> pl.LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create feature preprocessor
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Create data augmentation
    augmentors = build_agent_augmentor(cfg.data_augmentation) if 'data_augmentation' in cfg else None
    val_augmentors = build_agent_augmentor(cfg.val_data_augmentation) if 'val_data_augmentation' in cfg else None

    # Build dataset scenarios
    scenarios = build_scenarios(cfg, worker, model)

    # Create datamodule
    datamodule: pl.LightningDataModule = DataModule(
        feature_preprocessor=feature_preprocessor,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        augmentors=augmentors,
        val_augmentors=val_augmentors,
        worker=worker,
        scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_lightning_module(
    cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper
) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Build loss
    objectives = build_objectives(cfg)

    # Build metrics to evaluate the performance of predictions
    metrics = build_training_metrics(cfg)
    
    # Build aggregated metrics to evaluate the performance of predictions
    aggregated_metrics = build_aggregated_metrics(cfg) if "aggregated_metric" in cfg else {}

    # Create the complete Module
    if isinstance(cfg.checkpoint, str) and Path(cfg.checkpoint).is_file():
        logger.info(f"Loading checkpoint from {cfg.checkpoint}")
        model = LightningModuleWrapperCloseloop.load_from_checkpoint(
            cfg.checkpoint,
            strict=False,
            model=torch_module_wrapper,
            objectives=objectives,
            metrics=metrics,
            aggregated_metrics=aggregated_metrics,
            batch_size=cfg.data_loader.params.batch_size,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler if "lr_scheduler" in cfg else None,
            warm_up_lr_scheduler=cfg.warm_up_lr_scheduler
            if "warm_up_lr_scheduler" in cfg
            else None,
            objective_aggregate_mode=cfg.objective_aggregate_mode,
        )
    else:
        model = LightningModuleWrapperCloseloop(
            model=torch_module_wrapper,
            objectives=objectives,
            metrics=metrics,
            aggregated_metrics=aggregated_metrics,
            batch_size=cfg.data_loader.params.batch_size,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler if "lr_scheduler" in cfg else None,
            warm_up_lr_scheduler=cfg.warm_up_lr_scheduler
            if "warm_up_lr_scheduler" in cfg
            else None,
            objective_aggregate_mode=cfg.objective_aggregate_mode,
        )

    return cast(pl.LightningModule, model)

def _build_strategy(trainer_params: OmegaConf) -> Strategy:
    strat_name = trainer_params.strategy
    if trainer_params.devices == "auto" or trainer_params.devices == -1:
        num_devices = torch.cuda.device_count()
    else:
        num_devices = trainer_params.devices
    gpus = [torch.device(f"cuda:{i}") for i in range(num_devices)]
    if strat_name == "ddp":
        return DDPStrategy(
            accelerator="gpu", parallel_devices=gpus, find_unused_parameters=True
        )
    elif strat_name == "ray":
        return RayStrategy(
            num_workers=4,
            num_cpus_per_worker=20,
            use_gpu=True,
            resources_per_worker={"GPU": 2},
        )
    else:
        raise ValueError(
            f"Unknown or unsupported strategy: {strat_name}. Supported are ddp, ray."
        )


def build_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    callbacks = build_callbacks(cfg)

    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version="",
            prefix="",
        ),
    ]

    strategy = _build_strategy(params)
    del params.strategy
    del params.accelerator
    del params.devices

    if cfg.lightning.trainer.overfitting.enable:
        OmegaConf.set_struct(cfg, False)
        params = OmegaConf.merge(params, cfg.lightning.trainer.overfitting.params)
        params.check_val_every_n_epoch = params.max_epochs + 1
        OmegaConf.set_struct(cfg, True)

        return pl.Trainer(**params)

    if cfg.lightning.trainer.checkpoint.resume_training:
        OmegaConf.set_struct(cfg, False)
        if isinstance(cfg.lightning.trainer.checkpoint.resume_training, bool):
            # Resume training from latest checkpoint
            output_dir = Path(cfg.output_dir)
            date_format = cfg.date_format

            last_checkpoint = extract_last_checkpoint_from_experiment(
                output_dir, date_format
            )
            if not last_checkpoint:
                raise ValueError(
                    "Resume Training is enabled but no checkpoint was found!"
                )

            params.resume_from_checkpoint = str(last_checkpoint)
            latest_epoch = torch.load(last_checkpoint)["epoch"]
            params.max_epochs += latest_epoch
            logger.info(
                f"Resuming at epoch {latest_epoch} from checkpoint {last_checkpoint}"
            )

        else:
            # Resume training from designated checkpoint
            params.resume_from_checkpoint = str(
                cfg.lightning.trainer.checkpoint.resume_training
            )
            latest_epoch = torch.load(params.resume_from_checkpoint)["epoch"]
            params.max_epochs += latest_epoch
            logger.info(
                f"Resuming at epoch {latest_epoch} from checkpoint {params.resume_from_checkpoint}"
            )
        OmegaConf.set_struct(cfg, True)

    trainer = pl.Trainer(
        callbacks=callbacks,
        strategy=strategy,
        logger=loggers,
        **params,
    )

    return trainer
