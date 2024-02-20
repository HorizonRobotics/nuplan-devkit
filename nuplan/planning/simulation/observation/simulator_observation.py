from collections import defaultdict
from functools import reduce
import os
from typing import Dict, List, Optional, Type
import yaml
import csv
import pickle
from pyquaternion import Quaternion
import numpy as np
import torch
from torchvision.transforms.functional import resize, crop, to_pil_image
import cv2
import PIL.Image as PilImage

from nuplan.planning.simulation.observation.simulator.datasets.dynamic import rescale_K
from nuplan.planning.simulation.observation.simulator.pipelines.ogl import get_net, get_texture
from nuplan.planning.simulation.observation.simulator.utils.train import load_model_checkpoint, to_device, to_numpy
from nuplan.planning.simulation.observation.simulator.models.compose import NetAndTexture
from nuplan.planning.simulation.observation.simulator.gl.utils import load_scene_data_minimal, get_proj_matrix, setup_scene
from nuplan.planning.simulation.observation.simulator.gl.dataset import parse_input_string
from nuplan.planning.simulation.observation.simulator.gl.myrender import MyRenderSimple
from nuplan.planning.simulation.observation.simulator.gl.render import OffscreenRender
from nuplan.planning.simulation.observation.simulator.gl.programs import NNScene

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import Observation, SensorsWithTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from nuplan.common.actor_state.ego_state import EgoState

class SimulatorObservation(AbstractObservation):
    """
    Simulate agents based on trained log simulator.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
    ):
        """
        Constructor for SimulatorObservation.

        :param scenario: scenario
        :param sim_args: simulator args
        """
        self.current_iteration = 0
        self.scenario = scenario

    def load(self):
        # self.scene = NNScene()
        # setup_scene(self.scene, self.scene_data, use_mesh=False)
        pass

    def unload(self):
        # self.scene.delete()
        # self.scene = None
        pass

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return SensorsWithTracks  # type: ignore

    def initialize(self) -> None:
        # self.target_cameras = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_L2', 'CAM_B0', 'CAM_R2']
        # return

        csv_path = '/mnt/nas20/siqi01.chai/hoplan-simulation/test-set.csv'
        # csv_path = '/mnt/nas20/siqi01.chai/hoplan-simulation/train-300.csv'
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # enable for test set rendering, disable for training set rendering
            csv_files = [[*row, idx] for idx, row in enumerate(csv_reader)]
            csv_files = {'{}-{}'.format(row[0], row[3]) : row[-1] for row in csv_files}
        simulator_index = csv_files['{}-{}'.format(self.scenario._log_file_load_path.split('/')[-1], self.scenario._initial_lidar_timestamp)]
        simulator_model_path = '/mnt/nas20/siqi01.chai/test-scenes/test-scene-{}'.format(simulator_index)
        # simulator_model_path = '/mnt/nas20/siqi01.chai/train-scenes-300/train-scene-{}'.format(simulator_index)
        self.render_out_dir = '/mnt/nas26/siqi01.chai/sim-renders-stg2/scene-{}'.format(simulator_index)
        os.makedirs(self.render_out_dir, exist_ok=True)

        """Inherited, see superclass."""
        self.ego_state = None
        # target_cameras = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']
        self.target_cameras = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_L2', 'CAM_B0', 'CAM_R2']
        with open('{}/camera_params.pkl'.format(simulator_model_path), 'rb') as f:
            camera_params = pickle.load(f)
            self.camera_ego2cameras = []
            self.camera_intrinsics = []
            for cn in self.target_cameras:
                self.camera_ego2cameras.append(camera_params[cn]['ego2camera'])
                self.camera_intrinsics.append(camera_params[cn]['intrinsics'])
        
        scene_data = load_scene_data_minimal(simulator_model_path)
        net_ckpt = '{}/model-new/checkpoints/UNet_stage_0_epoch_23_net.pth'.format(simulator_model_path)
        tex_ckpt = '{}/model-new/checkpoints/PointTexture_stage_0_epoch_23.pth'.format(simulator_model_path)
        net = get_net()
        textures = {}
        assert scene_data['pointcloud'] is not None, 'set pointcloud'
        size = scene_data['pointcloud']['xyz'].shape[0]
        textures[0] = get_texture(8, size)
        net = load_model_checkpoint(net_ckpt, net)
        textures[0] = load_model_checkpoint(tex_ckpt, textures[0])
        self.model = NetAndTexture(net, textures)
        self.model.load_textures([0])
        self.model._textures[0].texture_.requires_grad = False

        self.input_format = 'uv_1d_p1, uv_1d_p1_ds1, uv_1d_p1_ds2, uv_1d_p1_ds3, uv_1d_p1_ds4'
        old_size = [1920, 1080]
        self.src_sh = np.array(old_size)
        # out_tgt_sh = [1920, 1072]
        # out_tgt_sh = [480, 256]
        # out_tgt_sh = [720, 384]
        out_tgt_sh = [960, 512]
        self.tgt_sh = np.array(list(map(lambda x:x//2**4 * 2**4, out_tgt_sh)))
        # self.tgt_sh = np.array(list(map(lambda x:x//2**4 * 2**4, self.src_sh // 4)))
        # self.tgt_sh = np.array(list(map(lambda x:x//2**4 * 2**4, src_sh)))
        self.all_img_rec = scene_data['all_img_rec']
        self.intrinsics_list = scene_data['intrinsic_matrix']
        self.view_list = scene_data['view_matrix']
        self.moving_instances = scene_data['moving_instances']
        
        def get_transform_matrix(translation, rotation, forward):
            t_mat = np.eye(4)
            t_mat[:3, -1] = translation
            r_mat = np.eye(4)
            r_mat[:3, :3] = rotation.rotation_matrix
            if forward:
                # first translate points, then rotate points
                return r_mat @ t_mat
            # first rotate points, thentranslate points
            return t_mat @ r_mat
        
        trk_recs = self.moving_instances.keys()
        for trk_rec in trk_recs:
            self.moving_instances[trk_rec]['tmat'] = get_transform_matrix(
                translation=-self.moving_instances[trk_rec]['ref_translation_world'],
                rotation=self.moving_instances[trk_rec]['ref_rotation_world'].inverse,
                forward=True,
                )
        
        self.origin_pose = scene_data['origin_pose']
        self.pcd = np.array(scene_data['pointcloud']['xyz'])

        self.renderer = MyRenderSimple(self.tgt_sh, self.input_format)    
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        self.model.eval()
        self.model.cuda()


    def get_observation(self, ego_state: EgoState) -> SensorsWithTracks:
        # imgs = torch.rand(6, 3, 224, 480)
        # rendered_imgs = {}
        # for img_idx, img in enumerate(imgs):
        #     r_img = to_pil_image(img)
        #     rendered_imgs[self.target_cameras[img_idx]] = r_img
        # print(self.current_iteration)
        # curr_tracks = self.scenario.get_tracked_objects_at_iteration(self.current_iteration).tracked_objects
        # return SensorsWithTracks(pointcloud=None, images=None, tracked_objects=curr_tracks)

        my_world2origin = np.array(self.origin_pose['world2origin'])
        my_origin2world = np.array(self.origin_pose['origin2world'])

        ego2world = ego_state.rear_axle.as_matrix_3d()  # with StateSE2, this matrix lacks z value, roll, and pitch
        # ego2world = self.scenario.get_3d_ego_transform_at_iteration(iteration=self.current_iteration)
        # heading = Quaternion(ego2world[-4:]).yaw_pitch_roll[0]
        # ego2world = np.array(
        #     [
        #         [np.cos(heading), -np.sin(heading), 0.0, ego2world[0]],
        #         [np.sin(heading), np.cos(heading), 0.0, ego2world[1]],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # )


        ego_log = self.scenario.get_3d_ego_transform_at_iteration(iteration=self.current_iteration)
        q = Quaternion(ego_log[-4:])   # qw, qx, qy, qz
        _, pitch, roll = q.yaw_pitch_roll
        pitch_m = np.array(
            [
                [np.cos(pitch), 0.0, np.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch), 0.0, np.cos(pitch)],
            ])
        roll_m = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll), -np.sin(roll)],
                [0.0, np.sin(roll), np.cos(roll)],
            ])
        ego2world[:3, :3] = reduce(np.dot, [ego2world[:3, :3], pitch_m, roll_m])
        ego2world[2, 3] = ego_log[2]
        world2ego = np.linalg.inv(ego2world)
        
        batch_view_matrix = []
        batch_intrinsic_matrix = []
        batch_proj_matrix = []
        for extr, intr in zip(self.camera_ego2cameras, self.camera_intrinsics):
            vm = reduce(np.dot, [extr, world2ego, my_origin2world])
            vm = np.linalg.inv(vm)
            vm[:, 1:3] *= -1
            assert vm.dtype == np.float64
            assert intr.dtype == np.float64
            znear, zfar = 0.1, 1000
            K = rescale_K(intr, self.tgt_sh/self.src_sh, True)
            pm = get_proj_matrix(K, self.tgt_sh, znear, zfar).astype(np.float32)
            batch_view_matrix.append(vm)
            batch_intrinsic_matrix.append(K)
            batch_proj_matrix.append(pm)
        batch_view_matrix = np.stack(batch_view_matrix, axis=0)
        batch_intrinsic_matrix = np.stack(batch_intrinsic_matrix, axis=0)
        batch_proj_matrix = np.stack(batch_proj_matrix, axis=0)

        ################################################################
        # calculate point flow for moving instances
        perturb = np.zeros_like(self.pcd)
        tracked_intances = self.scenario.get_3d_tracked_objects_at_iteration(iteration=self.current_iteration)
        tracked_intances = tracked_intances.tracked_objects.tracked_objects
        for trk_ins in tracked_intances:
            trk_token =  trk_ins.metadata.track_token
            if not trk_token in self.moving_instances:
                continue
            
            ins_target2world = trk_ins.box.center.as_matrix_3d()
            ins_world2ref = np.array(self.moving_instances[trk_token]['tmat'])
            total_m = reduce(np.dot, [my_world2origin, ins_target2world, ins_world2ref, my_origin2world])
            
            start_ids = self.moving_instances[trk_token]['dynamic_start_ids']
            end_ids = self.moving_instances[trk_token]['dynamic_end_ids']
            for s,e in zip(start_ids, end_ids):
                pts = self.pcd[s:e, :]
                pts = pts.T
                delta = total_m[:3, :3] @ pts + total_m[:3, -1:] - pts
                perturb[s:e, :] = delta.T
        ################################################################
        input_data = {'input': {'id': [0,0,0,0,0,0]},
                    'view_matrix': batch_view_matrix,
                    'intrinsic_matrix': batch_intrinsic_matrix,
                    'proj_matrix': batch_proj_matrix,
                    'points': self.pcd + perturb,
                }

        # rendered_imgs = {} 
        # for cam_i in range(6):
        #     with torch.set_grad_enabled(False):    
        #         input_data_i = {'input': {'id': [0]},
        #             'view_matrix': batch_view_matrix[cam_i : cam_i + 1],
        #             'intrinsic_matrix': batch_intrinsic_matrix[cam_i : cam_i + 1],
        #             'proj_matrix': batch_proj_matrix[cam_i : cam_i + 1],
        #             'points': self.pcd + perturb,
        #         }
        #         input_data_i['input'], depths = self.renderer.render(input_data_i)
        #         model_input = to_device(input_data_i['input'], 'cuda:0')
        #         outs = self.model(model_input)
        #         # print('rendered shape:', outs['im_out'].shape)

        #     r_img = outs['im_out'][0]
        #     r_img_copy = to_pil_image(torch.clamp(r_img, 0.0, 1.0))
        #     r_img_copy.save('simulator-rendered-res/simulator-rendered-image-{}-{}-{}-pil.jpg'.format(self.scenario.token, self.current_iteration, cam_i))
        #     r_img = resize(r_img, size=(256, 480))
        #     r_img = crop(r_img, top=32, left=0, height=224, width=480)
        #     r_img = torch.clamp(r_img, 0.0, 1.0)

        #     rendered_imgs[self.target_cameras[cam_i]] = to_pil_image(r_img)
        #     del r_img

        with torch.no_grad():    
            input_data['input'], depths = self.renderer.render(input_data)
            model_input = to_device(input_data['input'], 'cuda:0')
            outs = self.model(model_input)
        imgs = outs['im_out']
        imgs = resize(imgs, size=(256, 480))
        imgs = crop(imgs, top=32, left=0, height=224, width=480)
        imgs = torch.clamp(imgs, 0.0, 1.0)
        # imgs = imgs.clone().detach().cpu()
        rendered_imgs = {}
        for img_idx, img in enumerate(imgs):
            r_img = to_pil_image(img)
            rendered_imgs[self.target_cameras[img_idx]] = r_img
            # r_img.save('simulator-rendered-res/simulator-rendered-image-{}-{}-{}-pil.jpg'.format(self.scenario.token, self.current_iteration, img_idx))
            if not img_idx == 1:
                continue
            r_img.save('{}/rend-{}.jpg'.format(self.render_out_dir, self.current_iteration))
        # return Sensors(pointcloud=None, images=rendered_imgs)

        curr_tracks = self.scenario.get_tracked_objects_at_iteration(self.current_iteration).tracked_objects

        return SensorsWithTracks(pointcloud=None, images=rendered_imgs, tracked_objects=curr_tracks)


    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
        self.ego_state = history.current_state