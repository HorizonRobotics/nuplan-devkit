from typing import Any, List, Optional

import numpy as np
import os
import pytorch_lightning as pl
import torch
from collections import defaultdict
import torch.utils.data
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nuplan.common.geometry.compute import principal_value
from nuplan.planning.training.callbacks.utils.visualization_utils import \
    get_raster_with_trajectories_as_rgb
from nuplan_extent.planning.training.closed_loop.utils.numpy_util import \
    np_matrix_from_pose
import cv2
import imageio

class ClosedLoopCallback(pl.Callback):
    """Callback used to monitor the status of closed-loop training."""

    def __init__(
        self, 
        frequency: int=10, 
        **kwargs
    ): 
        """
        Initialize the callback

        :param frequency: logging frequency in iterations, defaults to 10
        """
        self.frequency = frequency
        self.pixel_size = 0.5
        self.canvas_rot_mat = np.array(np.array([
            [np.cos(np.pi/2), -np.sin(np.pi/2), 0.],
            [np.sin(np.pi/2), np.cos(np.pi/2), 0.],
            [0., 0., 1.]
        ]))
        # self.token2pics = defaultdict(list)
    
    def _log_training_images(self, logger, batch: Any, predictions: STEP_OUTPUT, global_index: int):
        """Log training images and predictions."""
        features, targets = batch[:2]
        images = []
        rasters = features["raster"]
        rasters.data = rasters.data.detach().cpu()
        gts = targets["trajectory"]
        gts.data = gts.data.detach().cpu()
        preds = predictions["trajectory"].to_device(torch.device('cpu'))
        preds.data = preds.data.detach().cpu()

        for raster, gt, pred in zip(rasters.unpack(), gts.unpack(), preds.unpack()):
            image = get_raster_with_trajectories_as_rgb(
                raster,
                gt,
                pred,
                pixel_size=self.pixel_size
            )
            images.append(image)
        image_batch = np.array(images)
        logger.add_images(
            "train/visualization",
            img_tensor=torch.from_numpy(image_batch),
            global_step=global_index,
            dataformats='NHWC',
        )

    def _save_closed_loop_scenario_videos(self, batch: Any, predictions: STEP_OUTPUT):
        features, targets, scenarios = batch
        scenario_tokens = [i.token for i in scenarios]

        rasters = features["raster"]
        rasters.data = rasters.data.detach().cpu()
        gts = targets["trajectory"]
        gts.data = gts.data.detach().cpu()
        preds = predictions["trajectory"].to_device(torch.device('cpu'))
        preds.data = preds.data.detach().cpu()

        for raster, gt, pred, token in zip(rasters.unpack(), gts.unpack(), preds.unpack(), scenario_tokens):
            image = get_raster_with_trajectories_as_rgb(
                raster,
                gt,
                pred,
                pixel_size=self.pixel_size
            )
            self.token2pics[token].append(image)

    def _log_training_targets(self, logger, batch: Any, outputs: STEP_OUTPUT, global_index: int):
        """Draw training targets before/after closed loop update."""
        # cl_targets = batch[1]['trajectory'].data.cpu().numpy()
        cl_targets = outputs["computed_traj"].cpu().numpy()
        gt_targets = outputs["original_targets"].cpu().numpy()
        rel_poses = outputs["relative_poses"].cpu().numpy()

        fig = plt.figure(figsize=(15, 15))
        
        for i in range(16):
            plt.subplot(4, 4, i+1) 

            # Rotate to canvas coordinate
            gt_pose_i = self.canvas_rot_mat[None, ...] @ np_matrix_from_pose(gt_targets[i])
            cl_pose_i = self.canvas_rot_mat[None, ...] @ rel_poses[i][None, ...] @ np_matrix_from_pose(cl_targets[i])

            # Draw gt position (0, 0) and closed-loop position using rel_poses
            x = (self.canvas_rot_mat @ rel_poses[i])[0, 2]
            y = (self.canvas_rot_mat @ rel_poses[i])[1, 2]
            plt.plot(0., 0., 'bo')
            plt.plot(x, y, 'ro')

            # Draw original target
            gt_x = gt_pose_i[:, 0, 2]
            gt_y = gt_pose_i[:, 1, 2]
            gt_sin = gt_pose_i[:, 1, 0]
            gt_cos = gt_pose_i[:, 0, 0]
            plt.plot(gt_x, gt_y, 'o--')
            # for xi, yi, sin, cos in zip(gt_x, gt_y, gt_sin, gt_cos):
            #     plt.arrow(xi, yi, cos*3, sin*3, head_length=0.7, head_width=0.02)

            # Draw closed-loop udpated target
            cl_x = cl_pose_i[:, 0, 2]
            cl_y = cl_pose_i[:, 1, 2]
            #
            #
            plt.plot(cl_x, cl_y, 'x--')
            h = np.arctan2(rel_poses[i, 1, 0], rel_poses[i, 0, 0]) * 180 / np.pi
            plt.title(f"x:{rel_poses[i, 0, 2]:.2f} | y:{rel_poses[i, 1, 2]:.2f} | h:{h:.2f}")
            plt.axis('equal')
            plt.grid('minor')
            # plt.xlim([-10., 10.])
            # plt.ylim([-2., 18.])
        
        logger.add_figure("train/targets", fig, global_step=global_index)

    def _log_scalar_values(self, logger, outputs: STEP_OUTPUT, pl_module: 'pl.LightningModule', step: int):
        """Log scalar values to tensorboard.
        
        The following values are logged:
        1) batch corrections: number of forced corrections per training batch.
        2) average reset count: the average number of resets a scenario has gone through.
        3) average iterations without reset: the average nubmer of timesteps before a reset has taken place.

        Args:
            logger: tensorboard logger.
            outputs (STEP_OUTPUT): model outputs.
            pl_module (pl.LightningModule): the lightningmodule during training.
            step (int): global step. Usually should be trainer.global_step.
        """
        if "batch_corrections" in outputs:
            logger.add_scalar("closed_loop/batch_corrections", outputs["batch_corrections"], global_step=step)
        num_resets, num_samples_reset, total_length_without_reset = 0, 0, 0
        for training_state in pl_module._token2state.values():
            num_resets += training_state.forced_reset_count
            num_samples_reset += int(training_state.forced_reset_count!=0)
            total_length_without_reset = total_length_without_reset + training_state.num_iter_without_reset
        avg_reset_count = 0 if num_samples_reset == 0 else num_resets / num_samples_reset
        avg_iter_without_reset = total_length_without_reset / max(len(pl_module._token2state), 1)
        logger.add_scalar("closed_loop/avg_reset_count", avg_reset_count, global_step=step)
        logger.add_scalar("closed_loop/avg_iter_without_reset", avg_iter_without_reset, global_step=step)

    def on_train_batch_end(
        self, 
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Visualize training examples or log training stats at the end of a batch."""
        if batch_idx % self.frequency == 0:
            assert hasattr(pl_module, "_token2state") and pl_module._token2state is not None
            assert hasattr(trainer, "global_step")

            logger = trainer.logger.experiment

            if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
                self._log_training_images(logger, batch, outputs, trainer.global_step)
                # self._log_training_targets(logger, batch, outputs, trainer.global_step)
                # self._log_scalar_values(logger, outputs, pl_module, trainer.global_step)

        # self._save_closed_loop_scenario_videos(batch, outputs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        os.makedirs("webm", exist_ok=True)
        for key, value in self.token2pics.items():
            video_writer = imageio.get_writer(f"webm/{key}.webm", fps=10, format="WEBM", mode="I", codec='vp9')

            for img in value:
                img_array = (img * 255).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

                video_writer.append_data(img_rgb)

            video_writer.close()
