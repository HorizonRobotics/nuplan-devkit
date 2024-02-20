from collections import defaultdict
from functools import reduce
import os
from typing import Dict, List, Optional, Type
import yaml
import csv
import pickle
from pyquaternion import Quaternion
import numpy as np
from scipy.ndimage import binary_opening, binary_closing
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, crop, to_pil_image
import cv2
import PIL.Image as PilImage
import pyrender
import trimesh

import time

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



def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=5.0),
            matrix=matrix
        ))

    return nodes


def map_new_frame_v2(M):
    """
    given a nuplan transform matrix, return an OpenGL transform matrix

    nuplan - OpenGL
    +x => -z
    +y => -x
    +z => +y
    (1,0,0) => (0,0,-1) => col 1
    (0,1,0) => (-1,0,0) => col 2
    (0,0,1) => (0,1,0) => col 3
    """

    # 1. nuplan point to OGL point
    s_nuplan_ogl = np.array([
        [0,-1,0,0],
        [0,0,1,0],
        [-1,0,0,0],
        [0,0,0,1],
        ])
    
    # 2. OGL point to nuplan point
    s_ogl_nuplan = np.linalg.inv(s_nuplan_ogl)

    M_new = s_nuplan_ogl @ M @ s_ogl_nuplan
    return M_new


def decode_camera_pose(VM):
    """
    given a nuplan view matrix, which maps a point from nuplan reference frame
    into a posed camera reference frame. We decode the camera pose matrix.
    The pose matrix is defined as [R @ T] in the nuplan reference frame

    nuplan camera - nuplan
    +x => -y
    +y => -z
    +z => +x
    (1,0,0) => (0,-1,0) => col 1
    (0,1,0) => (0,0,-1) => col 2
    (0,0,1) => (1,0,0) => col 3
    """

    # 1. nuplan point to nuplan camera point
    s_camera_nuplan = np.array([
        [0,0,1,0],
        [-1,0,0,0],
        [0,-1,0,0],
        [0,0,0,1],
        ])
    VM_new = s_camera_nuplan @ VM

    # 2. view matrix to pose matrix
    R = VM_new[:3, :3]
    t = VM_new[:3, -1]
    R_inv = np.transpose(R)
    t_inv = -np.dot(R_inv, t)
    pose = np.eye(4)
    pose[:3, :3] = R_inv
    pose[:3, -1] = t_inv
    return pose


def apply_custom_kernel_pytorch_fixed_v2(image, kernel_size=15):
    """
    Apply a custom kernel to the image using PyTorch for parallel processing with fixed mask application.
    """
    # Convert the numpy image to a PyTorch tensor and add a batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Define the colors as tensors
    red_color = torch.tensor([255, 0, 0], dtype=torch.float32)

    # Create kernels to count white and red pixels
    red_kernel = (image_tensor == red_color.view(3, 1, 1)).all(dim=1, keepdim=True).type(torch.float32)

    # Convolve the kernels over the image to count colors in each patch
    red_counts = F.conv2d(red_kernel, torch.ones(1, 1, kernel_size, kernel_size), padding=kernel_size//2)

    # Determine the dominant color in each patch and create masks
    red_dominant = red_counts > 75

    image_tensor[:, 0, :, :][red_dominant.squeeze(1)] = red_color[0]
    image_tensor[:, 1, :, :][red_dominant.squeeze(1)] = red_color[1]
    image_tensor[:, 2, :, :][red_dominant.squeeze(1)] = red_color[2]

    # Convert the tensor back to numpy format
    modified_image = image_tensor.squeeze(0).permute(1, 2, 0).byte().numpy()

    return modified_image


class SimulatorGraphicsObservation(AbstractObservation):
    """
    Simulate agents based on trained log simulator.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
    ):
        """
        Constructor for SimulatorGraphicsObservation.

        :param scenario: scenario
        :param sim_args: simulator args
        """
        self.current_iteration = 0
        self.scenario = scenario

    def load(self):
        self.scene = NNScene()
        setup_scene(self.scene, self.scene_data, use_mesh=False)

    def unload(self):
        self.scene.delete()
        self.scene = None

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return SensorsWithTracks  # type: ignore

    def initialize(self) -> None:
        # csv_path = '/mnt/nas20/siqi01.chai/hoplan-simulation/test-set.csv'
        csv_path = '/mnt/nas20/siqi01.chai/hoplan-simulation/train-300.csv'
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # next(csv_reader)
            csv_files = [[*row, idx] for idx, row in enumerate(csv_reader)]
            csv_files = {'{}-{}'.format(row[0], row[3]) : row[-1] for row in csv_files}
        simulator_index = csv_files['{}-{}'.format(self.scenario._log_file_load_path.split('/')[-1], self.scenario._initial_lidar_timestamp)]
        # simulator_model_path = '/mnt/nas20/siqi01.chai/test-scenes/test-scene-{}'.format(simulator_index)
        simulator_model_path = '/mnt/nas20/siqi01.chai/train-scenes-300/train-scene-{}'.format(simulator_index)
        # self.render_out_dir = '/mnt/nas20/siqi01.chai/sim-renders/online-render-tmp-2/scene-{}'.format(simulator_index)
        self.render_out_dir = '/mnt/nas26/siqi01.chai/sim-renders/online-render-graphics/scene-{}'.format(simulator_index)
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

        # import pytorch_lightning as pl
        # class MyLightningModule(pl.LightningModule):
        #     def __init__(self, model):
        #         super(MyLightningModule, self).__init__()
        #         # Initialize your PyTorch model here
        #         self.model = model  # Replace MyModel with your model class
        # model_wrapper = MyLightningModule(self.model)
        # self.model = model_wrapper.model

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


        # pyrender: mesh renderer part
        mesh_model = trimesh.load('./objaverse-cars/mycar3.glb')
        car_mesh = trimesh.util.concatenate(list(mesh_model.geometry.values()))
        rescale = 0.8
        tform = np.asarray([
            -(car_mesh.bounds[1][i] + car_mesh.bounds[0][i])/2.
            for i in range(3)
        ])
        matrix = np.eye(4)
        matrix[:3, 3] = tform
        car_mesh.apply_transform(matrix)
        matrix = np.eye(4)
        matrix[:3, :3] *= rescale
        car_mesh.apply_transform(matrix)
        matrix = np.eye(4)
        matrix[:3, 3] = -tform
        car_mesh.apply_transform(matrix)
        car_mesh = pyrender.Mesh.from_trimesh(car_mesh)
        self.car_mesh_scene = pyrender.Scene()
        self.car_mesh_node = self.car_mesh_scene.add(car_mesh)
        light_nodes = create_raymond_lights()
        for light_node in light_nodes:
            self.car_mesh_scene.add_node(light_node, parent_node=self.car_mesh_node)
        rescale_intr = rescale_K(self.camera_intrinsics[0], self.tgt_sh/self.src_sh, True)
        camera = pyrender.IntrinsicsCamera(rescale_intr[0, 0], rescale_intr[1, 1], rescale_intr[0, 2], rescale_intr[1, 2])
        self.car_mesh_camera_node = self.car_mesh_scene.add(camera)
        self.car_mesh_renderer = pyrender.OffscreenRenderer(self.tgt_sh[0], self.tgt_sh[1])
        self.mesh_instance_id = 2
        print('number of mov ins: ', len(self.moving_instances))
        self.mesh_point_ids = None
        mesh_out_dir = os.path.join('/mnt/nas26/siqi01.chai/mesh-exp/', 'scene-{}'.format(simulator_index))
        os.makedirs(mesh_out_dir, exist_ok=True)

        #################
        # model-raw | graphics-raw
        # pc-mask | depth mask
        # overlap | sparse-render
        # final-mask | final-render
        #################
        self.mesh_model_raw_dir = os.path.join(mesh_out_dir, 'model-raw')
        os.makedirs(self.mesh_model_raw_dir, exist_ok=True)

        self.mesh_graphics_raw_dir = os.path.join(mesh_out_dir, 'graphics-raw')
        os.makedirs(self.mesh_graphics_raw_dir, exist_ok=True)

        self.mesh_mask_pc_dir = os.path.join(mesh_out_dir, 'mask-pc')
        os.makedirs(self.mesh_mask_pc_dir, exist_ok=True)

        self.mesh_mask_dp_dir = os.path.join(mesh_out_dir, 'mask-dp')
        os.makedirs(self.mesh_mask_dp_dir, exist_ok=True)

        self.mesh_overlap_dir = os.path.join(mesh_out_dir, 'overlap')
        os.makedirs(self.mesh_overlap_dir, exist_ok=True)
        self.mesh_sparse_render_dir = os.path.join(mesh_out_dir, 'sparse-render')
        os.makedirs(self.mesh_sparse_render_dir, exist_ok=True)

        self.mesh_final_mask_dir = os.path.join(mesh_out_dir, 'final-mask')
        os.makedirs(self.mesh_final_mask_dir, exist_ok=True)
        self.mesh_final_render_dir = os.path.join(mesh_out_dir, 'final-render')
        os.makedirs(self.mesh_final_render_dir, exist_ok=True)


    def get_observation(self, ego_state: EgoState) -> SensorsWithTracks:
        # imgs = torch.rand(6, 3, 224, 480)
        # rendered_imgs = {}
        # for img_idx, img in enumerate(imgs):
        #     r_img = to_pil_image(img)
        #     rendered_imgs[self.target_cameras[img_idx]] = r_img
        # return Sensors(pointcloud=None, images=rendered_imgs)


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
        batch_mesh_camera_matrix = []
        for extr, intr in zip(self.camera_ego2cameras, self.camera_intrinsics):
            vm = reduce(np.dot, [extr, world2ego, my_origin2world])

            # calculate pose for mesh rendering
            mesh_camera_matrix = map_new_frame_v2(decode_camera_pose(vm))
            batch_mesh_camera_matrix.append(mesh_camera_matrix)

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

        with torch.set_grad_enabled(False):    
            input_data['input'], depths = self.renderer.render(input_data)
            model_input = to_device(input_data['input'], 'cuda:0')
            outs = self.model(model_input)
        imgs = outs['im_out']


        ################################################################
        # calculate pose for the mesh_instance_id car in the scene, we will replace that car with mesh
        mesh_colors = []
        mesh_depths = []
        mesh_trk_token = None
        if len(self.moving_instances) > self.mesh_instance_id:
            mesh_trk_token = list(self.moving_instances.keys())[self.mesh_instance_id]
            for trk_ins in tracked_intances: 
                if trk_ins.metadata.track_token == mesh_trk_token:
                    ins_target2world = trk_ins.box.center.as_matrix_3d()
                    ins_world2ref = np.array(self.moving_instances[mesh_trk_token]['tmat'])
                    mesh_total_m = reduce(np.dot, [my_world2origin, ins_target2world])
                    mesh_total_m = map_new_frame_v2(mesh_total_m)
                    default_align_m = np.array([
                        [-1,0,0,0], 
                        [0,0,1,0], 
                        [0,1,0,0], 
                        [0,0,0,1],
                        ])
                    mesh_total_m = mesh_total_m @ default_align_m
                    self.car_mesh_node.matrix = mesh_total_m

                    # print('mesh rendering')
                    for cam_i, cam_pose in enumerate(batch_mesh_camera_matrix):
                        if not cam_i == 1:
                            mesh_colors.append(None)
                            mesh_depths.append(None)
                            continue
                        self.car_mesh_camera_node.matrix = cam_pose
                        color, depth = self.car_mesh_renderer.render(self.car_mesh_scene)
                        mesh_colors.append(color)
                        mesh_depths.append(depth)
                        graphics_raw_out = os.path.join(self.mesh_graphics_raw_dir, '{}.png'.format(self.current_iteration))
                        cv2.imwrite(graphics_raw_out, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                    break

        # ################### combine with mesh #########################
        # # import pdb; pdb.set_trace()
        # tmp_imgs = imgs.clone().detach().cpu()
        # for img_idx, img in enumerate(tmp_imgs):
        #     if not mesh_colors:
        #         continue
        #     r_img = np.array(to_pil_image(img))
        #     depth_mask = mesh_depths[img_idx] > 0
        #     r_img[depth_mask] = mesh_colors[img_idx][depth_mask]
        #     if not img_idx == 1:
        #         continue
        #     cv2.imwrite('./mycar_mesh_render_combined/{}.png'.format(self.current_iteration), cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR))
        #     depth_mask = (depth_mask * 255).astype(np.uint8)
        #     cv2.imwrite('./mycar_mesh_render_mask/{}.png'.format(self.current_iteration), depth_mask)
        # ###############################################################
            
        ################### combine with mesh #########################
        if len(mesh_colors) > 0:
            tmp_depth_buffer = input_data['input']['uv_1d_p1'].clone().detach().cpu().numpy()
            tmp_imgs = torch.clamp(imgs, 0.0, 1.0).detach().cpu()

            if self.mesh_point_ids is None:
                start_ids = self.moving_instances[mesh_trk_token]['dynamic_start_ids']
                end_ids = self.moving_instances[mesh_trk_token]['dynamic_end_ids']
                mesh_point_ids = []
                for s,e in zip(start_ids, end_ids):
                    mesh_point_ids.extend([point_id for point_id in range(s, e)])
                self.mesh_point_ids = np.array(list(set(mesh_point_ids)))

            for img_idx, (img, depth_buffer) in enumerate(zip(tmp_imgs, tmp_depth_buffer)):
                if not img_idx == 1:
                    continue
                r_img = np.array(to_pil_image(img))
                depth_mask = mesh_depths[img_idx] > 0
                depth_mask = depth_mask == 1.0
                dbf = depth_buffer[0].ravel().astype(int)
                point_index_mask = np.isin(dbf, self.mesh_point_ids)
                point_index_mask = point_index_mask.reshape((512, 960))


                #################
                # model-raw | graphics-raw
                # pc-mask | depth mask
                # overlap | sparse-render
                # final-mask | final-render
                #################

                # model-raw out
                model_raw_path = os.path.join(self.mesh_model_raw_dir, '{}.png'.format(self.current_iteration))
                cv2.imwrite(model_raw_path, cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR))
                
                # depth mask
                depth_mask_out = (depth_mask * 255).astype(np.uint8)
                depth_mask_out_path = os.path.join(self.mesh_mask_dp_dir, '{}.png'.format(self.current_iteration))
                cv2.imwrite(depth_mask_out_path, depth_mask_out)
                
                # pc-mask
                pc_mask = np.zeros((512 * 960, 3))
                pc_mask[dbf > 0, :] = 255
                pc_mask[point_index_mask.ravel(), :] = np.array([255, 0, 0])
                pc_mask = pc_mask.reshape((512, 960, 3)).astype(np.uint8)
                mask_pc_out = os.path.join(self.mesh_mask_pc_dir, '{}.png'.format(self.current_iteration))
                cv2.imwrite(mask_pc_out, cv2.cvtColor(pc_mask, cv2.COLOR_RGB2BGR))
                
                # overlap
                overlap_img = np.array(r_img)
                overlap_img[depth_mask] = mesh_colors[img_idx][depth_mask]
                overlap_img_path = os.path.join(self.mesh_overlap_dir, '{}.png'.format(self.current_iteration))
                cv2.imwrite(overlap_img_path, cv2.cvtColor(overlap_img, cv2.COLOR_RGB2BGR))

                # render-sparse
                sparse_img = np.array(r_img)
                sparse_mask = np.logical_and(depth_mask, point_index_mask)
                sparse_img[sparse_mask] = mesh_colors[img_idx][sparse_mask]
                sparse_img_path = os.path.join(self.mesh_sparse_render_dir, '{}.png'.format(self.current_iteration))
                cv2.imwrite(sparse_img_path, cv2.cvtColor(sparse_img, cv2.COLOR_RGB2BGR))

                # final-mask
                # pc_mask_dense = apply_custom_kernel_pytorch_fixed_v2(pc_mask)
                pc_mask_dense = pc_mask
                # binary mask denoising and convex hull
                red_mask = np.all(pc_mask_dense == [255, 0, 0], axis=-1)
                red_mask = binary_closing(red_mask, structure=np.ones((7,7)))
                red_mask = binary_opening(red_mask, structure=np.ones((15,15)))
                pc_mask_dense_path = os.path.join(self.mesh_final_mask_dir, '{}.png'.format(self.current_iteration))
                # cv2.imwrite(pc_mask_dense_path, cv2.cvtColor(pc_mask_dense, cv2.COLOR_RGB2BGR))
                cv2.imwrite(pc_mask_dense_path,  (red_mask * 255).astype(np.uint8))

                # final image
                # red_mask = np.all(pc_mask_dense == [255, 0, 0], axis=-1)
                # red_rows, red_cols = np.where(red_mask)
                # if red_rows.size == 0 or red_cols.size == 0:
                #     final_mask = np.zeros_like(depth_mask).astype(bool)
                # else:
                #     min_row, max_row = red_rows.min(), red_rows.max()
                #     min_col, max_col = red_cols.min(), red_cols.max()

                #     final_mask = np.array(depth_mask)
                #     final_mask[:min_row, :] = False
                #     final_mask[max_row:, :] = False
                #     final_mask[:, :min_col] = False
                #     final_mask[:, max_col:] = False
                final_mask = np.logical_and(red_mask, depth_mask)

                final_img = np.array(r_img)
                final_img[final_mask] = mesh_colors[img_idx][final_mask]
                final_img_path = os.path.join(self.mesh_final_render_dir, '{}.png'.format(self.current_iteration))
                cv2.imwrite(final_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
                
        ###############################################################

        imgs = resize(imgs, size=(256, 480))
        imgs = crop(imgs, top=32, left=0, height=224, width=480)
        imgs = torch.clamp(imgs, 0.0, 1.0)
        rendered_imgs = {}
        for img_idx, img in enumerate(imgs):
            r_img = to_pil_image(img)
            rendered_imgs[self.target_cameras[img_idx]] = r_img
            if not img_idx == 1:
                continue
            # r_img.save('{}/rend-{}.jpg'.format(self.render_out_dir, self.current_iteration))
        curr_tracks = self.scenario.get_tracked_objects_at_iteration(self.current_iteration).tracked_objects
        return SensorsWithTracks(pointcloud=None, images=rendered_imgs, tracked_objects=curr_tracks)


    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
        self.ego_state = history.current_state