# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import json
import os.path
from enum import Enum
from functools import reduce
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R

import trimesh
import torchvision
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection, cameras_from_opencv_projection
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util
from configs.config import parse_args
from datasets.generate_dataset import VideoDataset
# from datasets.image_dataset import ImagesDataset
from face_detector import FaceDetector
from flame.FLAME import FLAME, FLAMETex
from image import tensor2im
from renderer import Renderer
from math import sqrt

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
rank = 42
torch.manual_seed(rank)
torch.cuda.manual_seed(rank)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(rank)
I = torch.eye(3)[None].cuda().detach()
I6D = matrix_to_rotation_6d(I)
mediapipe_idx = np.load('flame/mediapipe/mediapipe_landmark_embedding.npz', allow_pickle=True, encoding='latin1')['landmark_indices'].astype(int)
left_iris_flame = [4597, 4542, 4510, 4603, 4570]
right_iris_flame = [4051, 3996, 3964, 3932, 4028]
left_iris_mp = [468, 469, 470, 471, 472]
right_iris_mp = [473, 474, 475, 476, 477]



class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64
    SHAPE_MERGE = 128


class Tracker(object):
    def __init__(self, config, device='cuda:0'):
        self.config = config
        self.device = device
        self.face_detector = FaceDetector('google')
        self.pyr_levels = config.pyr_levels
        self.actor_name = self.config.config_name
        self.kernel_size = self.config.kernel_size
        self.sigma = None if self.config.sigma == -1 else self.config.sigma
        self.global_step = 0

        logger.add(os.path.join(self.config.save_folder, self.actor_name, 'train.log'))

        # Latter will be set up
        self.frame = 0
        self.is_initializing = False
        self.image_size = torch.tensor([[config.image_size[0], config.image_size[1]]]).cuda()
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.actor_name)
        self.checkpoint_folder = os.path.join(self.save_folder, self.actor_name, "checkpoint")
        self.input_folder = os.path.join(self.save_folder, self.actor_name, "input")
        self.pyramid_folder = os.path.join(self.save_folder, self.actor_name, "pyramid")
        self.mesh_folder = os.path.join(self.save_folder, self.actor_name, "mesh")
        self.depth_folder = os.path.join(self.save_folder, self.actor_name, "depth")
        self.create_output_folders()
        self.writer = SummaryWriter(log_dir=self.save_folder + self.actor_name + '/logs')
        self.setup_renderer()

    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()

    def create_output_folders(self):
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)
        Path(self.depth_folder).mkdir(parents=True, exist_ok=True)
        Path(self.mesh_folder).mkdir(parents=True, exist_ok=True)
        Path(self.input_folder).mkdir(parents=True, exist_ok=True)
        Path(self.pyramid_folder).mkdir(parents=True, exist_ok=True)

    def setup_renderer(self):
        mesh_file = '/netscratch/jeetmal/models/metrical-tracker/data/head_template_mesh.obj'
        self.config.image_size = self.get_image_size()
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)
        self.diff_renderer = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)
        self.faces = load_obj(mesh_file)[1]

        raster_settings = RasterizationSettings(
            image_size=self.get_image_size(),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )

    def load_checkpoint(self, idx=-1):
        if not os.path.exists(self.checkpoint_folder):
            return False
        snaps = sorted(glob(self.checkpoint_folder + '/*.frame'))
        if len(snaps) == 0:
            logger.info('Training from beginning...')
            return False
        if len(snaps) == len(self.dataset):
            logger.info('Training has finished...')
            exit(0)

        last_snap = snaps[idx]
        payload = torch.load(last_snap)

        camera_params = payload['camera']
        # self.R = torch.from_numpy(camera_params['R']).to(self.device)
        # self.t = torch.from_numpy(camera_params['t']).to(self.device)
        self.principal_point = torch.from_numpy(camera_params['pp']).to(self.device)
        
        flame_params = payload['flame']
        self.rotation = nn.Parameter(torch.from_numpy(flame_params['rotation']).to(self.device))
        self.translation = nn.Parameter(torch.from_numpy(flame_params['translation']).to(self.device))
        self.tex = nn.Parameter(torch.from_numpy(flame_params['tex']).to(self.device))
        self.expr = nn.Parameter(torch.from_numpy(flame_params['expr']).to(self.device))
        self.sh = nn.Parameter(torch.from_numpy(flame_params['sh']).to(self.device))
        self.shape = nn.Parameter(torch.from_numpy(flame_params['shape']).to(self.device))
        # self.mica_shape = nn.Parameter(torch.from_numpy(flame_params['shape']).to(self.device))
        self.eyes = nn.Parameter(torch.from_numpy(flame_params['eyes_pose']).to(self.device))
        self.eyelids = nn.Parameter(torch.from_numpy(flame_params['eyelids']).to(self.device))
        self.jaw = nn.Parameter(torch.from_numpy(flame_params['jaw_pose']).to(self.device))

        self.frame = int(payload['frame_id'])
        self.global_step = payload['global_step']
        self.update_prev_frame()
        self.image_size = torch.from_numpy(payload['img_size'])[None].to(self.device)
        self.setup_renderer()

        logger.info(f'Snapshot loaded for frame {self.frame}')

        return True

    def save_checkpoint(self, frame_id):
        opencv = opencv_from_cameras_projection(self.cameras, self.image_size)
        neck = np.ones((1, 3), dtype=np.float32)

        frame = {
            'flame': {
                'expr': self.expr.clone().detach().cpu().numpy(),
                'shape': self.shape.clone().detach().cpu().numpy(),
                'tex': self.tex.clone().detach().cpu().numpy(),
                'sh': self.sh.clone().detach().cpu().numpy(),
                'neck_pose': neck,
                'eyes_pose': self.eyes.clone().detach().cpu().numpy(),
                'eyelids': self.eyelids.clone().detach().cpu().numpy(),
                'jaw_pose': self.jaw.clone().detach().cpu().numpy(),
                'rotation': self.rotation.clone().detach().cpu().numpy(),
                'translation': self.translation.clone().detach().cpu().numpy()
            },
            'camera': {
                # 'R': self.R.clone().detach().cpu().numpy(),
                # 't': self.t.clone().detach().cpu().numpy(),
                'pp': self.principal_point.clone().detach().cpu().numpy()
            },
            'opencv': {
                'R': opencv[0].clone().detach().cpu().numpy(),
                't': opencv[1].clone().detach().cpu().numpy(),
                'K': opencv[2].clone().detach().cpu().numpy(),
            },
            'img_size': self.image_size.clone().detach().cpu().numpy()[0],
            'frame_id': frame_id,
            'global_step': self.global_step
        }

        vertices, _, _ = self.flame(
            shape_params=self.shape,
            expression_params=self.expr,
            rot_params=matrix_to_rotation_6d(self.rotation),
            trans_params=self.translation,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            eye_pose_params=self.eyes,
            jaw_pose_params=self.jaw,
            eyelid_params=self.eyelids
        )

        f = self.diff_renderer.faces[0].cpu().numpy()
        v = vertices[0].cpu().numpy()

        trimesh.Trimesh(faces=f, vertices=v, process=False).export(f'{self.mesh_folder}/{frame_id}.obj')
        torch.save(frame, f'{self.checkpoint_folder}/{frame_id}.frame')
        flame_param_path = self.checkpoint_folder.replace('checkpoint', 'flame_param')
        if not os.path.exists(flame_param_path):
            os.makedirs(flame_param_path)
        np.savez(f'{flame_param_path}/{frame_id}.npz', **frame['flame'])

    def save_canonical(self):
        canon = os.path.join(self.save_folder, self.actor_name, "canonical.obj")
        if not os.path.exists(canon):
            from scipy.spatial.transform import Rotation as R
            rotvec = np.zeros(3)
            rotvec[0] = 12.0 * np.pi / 180.0
            jaw = matrix_to_rotation_6d(torch.from_numpy(R.from_rotvec(rotvec).as_matrix())[None, ...].cuda()).float()
            vertices = self.flame(cameras=torch.inverse(self.cameras.R), shape_params=self.shape, jaw_pose_params=jaw)[0].detach()
            faces = self.diff_renderer.faces[0].cpu().numpy()
            trimesh.Trimesh(faces=faces, vertices=vertices[0].cpu().numpy(), process=False).export(canon)

    def get_heatmap(self, values):
        l2 = tensor2im(values)
        l2 = cv2.cvtColor(l2, cv2.COLOR_RGB2BGR)
        l2 = cv2.normalize(l2, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(l2, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(cv2.addWeighted(heatmap, 0.75, l2, 0.25, 0).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)

        return heatmap

    def update_prev_frame(self):
        self.prev_rotation = self.rotation.clone().detach()
        self.prev_translation = self.translation.clone().detach()
        self.prev_expr = self.expr.clone().detach()
        self.prev_eyes = self.eyes.clone().detach()
        self.prev_jaw = self.jaw.clone().detach()

    def render_shape(self, vertices, faces=None, white=True):
        B = vertices.shape[0]
        V = vertices.shape[1]
        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))

        fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_world, cameras=self.cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return rendering[:, 0:3, :, :]

    def to_cuda(self, batch, unsqueeze=False):
        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                if unsqueeze:
                    batch[key] = batch[key][None]

        return batch

    def create_parameters(self):
        bz = 1
        self.rotation = nn.Parameter(torch.eye(3).unsqueeze(0).to(self.device))
        self.translation = nn.Parameter(torch.zeros(bz, 3).float().to(self.device))
        # self.shape = nn.Parameter(self.mica_shape)
        self.shape = nn.Parameter(torch.zeros(bz, self.config.num_shape_params).float().to(self.device))
        # self.mica_shape = nn.Parameter(self.mica_shape)
        self.tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        self.expr = nn.Parameter(torch.zeros(bz, self.config.num_exp_params).float().to(self.device))
        self.sh = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1))
        self.jaw = nn.Parameter(matrix_to_rotation_6d(I))
        self.eyelids = nn.Parameter(torch.zeros(bz, 2).float().to(self.device))

    @staticmethod
    def save_tensor(tensor, path='tensor.jpg'):
        img = (tensor[0].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
        cv2.imwrite(path, img)

    def parse_mask(self, ops, batch, visualization=False):
        _, _, h, w = ops['alpha_images'].shape
        result = ops['mask_images_rendering']

        if visualization:
            result = ops['mask_images']

        return result.detach()

    def update(self, param_groups):
        for param in param_groups:
            for i, name in enumerate(param['name']):
                setattr(self, name, nn.Parameter(param['params'][i].clone().detach()))

    def get_param(self, name, param_groups):
        for param in param_groups:
            if name in param['name']:
                return param['params'][param['name'].index(name)]
        return getattr(self, name)

    def clone_params_tracking(self):
        params = [
            {'params': [nn.Parameter(self.expr.clone())], 'lr': 0.025, 'name': ['expr']},
            {'params': [nn.Parameter(self.eyes.clone())], 'lr': 0.001, 'name': ['eyes']},
            {'params': [nn.Parameter(self.eyelids.clone())], 'lr': 0.001, 'name': ['eyelids']},
            {'params': [nn.Parameter(self.rotation.clone())], 'lr': self.config.rotation_lr, 'name': ['rotation']},
            {'params': [nn.Parameter(self.translation.clone())], 'lr': self.config.translation_lr, 'name': ['translation']},
            {'params': [nn.Parameter(self.sh.clone())], 'lr': 0.001, 'name': ['sh']}
        ]

        if self.config.optimize_jaw:
            params.append({'params': [nn.Parameter(self.jaw.clone().detach())], 'lr': 0.001, 'name': ['jaw']})

        return params

    def clone_params_initialization(self):
        params = [
            {'params': [nn.Parameter(self.expr.clone())], 'lr': 0.025, 'name': ['expr']},
            {'params': [nn.Parameter(self.eyes.clone())], 'lr': 0.001, 'name': ['eyes']},
            {'params': [nn.Parameter(self.eyelids.clone())], 'lr': 0.01, 'name': ['eyelids']},
            {'params': [nn.Parameter(self.sh.clone())], 'lr': 0.01, 'name': ['sh']},
            {'params': [nn.Parameter(self.translation.clone())], 'lr': 0.05, 'name': ['translation']},
            {'params': [nn.Parameter(self.rotation.clone())], 'lr': 0.05, 'name': ['rotation']},
        ]

        if self.config.optimize_shape:
            params.append({'params': [nn.Parameter(self.shape.clone().detach())], 'lr': 0.025, 'name': ['shape']})

        if self.config.optimize_jaw:
            params.append({'params': [nn.Parameter(self.jaw.clone().detach())], 'lr': 0.001, 'name': ['jaw']})

        return params

    def clone_params_color(self):
        params = [
            {'params': [nn.Parameter(self.sh.clone())], 'lr': 0.05, 'name': ['sh']},
            {'params': [nn.Parameter(self.tex.clone())], 'lr': 0.05, 'name': ['tex']},
        ]

        return params

    @staticmethod
    def reduce_loss(losses):
        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return all_loss

    @staticmethod
    def repeat_n_times(x: torch.Tensor, n: int):
        """Expand a tensor from shape [F, ...] to [F*n, ...]"""
        return x.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, *x.shape[1:])

    def optimize_camera(self, batch, steps=5000):
        batch = self.to_cuda(batch)
        images, landmarks, landmarks_dense, lmk_dense_mask, lmk_mask = self.parse_batch(batch)
        
        h, w = images.shape[2:4]
        # self.shape = batch['shape']
        # self.mica_shape = batch['shape'].clone().detach()  # Save it for regularization

        # Important to initialize
        self.create_parameters()
        # Pytorch3d cameras
        self.cameras = batch['cameras'].to(self.device)
        self.principal_point = batch['principal_point'].to(self.device)

        params = [{'params': [self.translation], 'lr': 0.06, 'name': ['translation']},
                    {'params': [self.rotation], 'lr': 0.04, 'name': ['rotation']},]
        
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


        t = tqdm(range(steps), desc='', leave=True, miniters=100)
        for k in t:
            _, lmk68, lmkMP = self.flame(rot_params=matrix_to_rotation_6d(self.rotation), trans_params=self.translation, shape_params=self.shape, expression_params=self.expr, eye_pose_params=self.eyes, jaw_pose_params=self.jaw)
            lmk68_n = self.repeat_n_times(lmk68, self.num_cameras)
            lmkMP_n = self.repeat_n_times(lmkMP, self.num_cameras)

            points68 = self.cameras.transform_points_screen(lmk68_n)[..., :2] 
            pointsMP = self.cameras.transform_points_screen(lmkMP_n)[..., :2]
            
            losses = {}
            losses['pp_reg'] = torch.sum(self.principal_point ** 2)
            losses['lmk68'] = util.lmk_loss(points68, landmarks[..., :2], [h, w], lmk_mask) * self.config.w_lmks
            losses['lmkMP'] = util.lmk_loss(pointsMP, landmarks_dense[..., :2], [h, w], lmk_dense_mask) * self.config.w_lmks
            # losses['loss/lmk_oval'] = util.oval_lmk_loss(points68, landmarks, [h,w], lmk_mask) * self.config.w_lmks_oval


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()

            loss = all_loss.item()
            # self.writer.add_scalar('camera', loss, global_step=k)
            t.set_description(f'Loss for camera {loss:.4f}')
            self.frame += 1
            if k % 500 == 0 and k > 0:
                # pass
                self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE_OVERLAY]], frame_dst='/camera', save=False, dump_directly=True)

        self.frame = 0

    def optimize_color(self, batch, pyramid, params_func, pho_weight_func, reg_from_prev=False):
        self.update_prev_frame()
        images, landmarks, landmarks_dense, lmk_dense_mask, lmk_mask = self.parse_batch(batch)

        aspect_ratio = util.get_aspect_ratio(images)
        h, w = images.shape[2:4]
        logs = []

        for k, level in enumerate(pyramid):
            img, iters, size, image_size = level

            # Optimizer per step
            optimizer = torch.optim.Adam(params_func())
            params = optimizer.param_groups

            shape = self.get_param('shape', params)
            expr = self.get_param('expr', params)
            eyes = self.get_param('eyes', params)
            eyelids = self.get_param('eyelids', params)
            jaw = self.get_param('jaw', params)
            tex = self.get_param('tex', params)
            sh = self.get_param('sh', params)
            pp = self.principal_point
            if self.is_initializing:
                translation = self.translation
                rotation = self.rotation
            else:
                translation = self.get_param('translation', params)
                rotation = self.get_param('rotation', params)

            scale = image_size[0] / h
            self.diff_renderer.set_size(size)
            self.debug_renderer.rasterizer.raster_settings.image_size = size
            flipped = torch.flip(img, [2, 3])

            image_lmks68 = (landmarks * scale).to(self.device)
            image_lmksMP = (landmarks_dense * scale).to(self.device)
            left_iris = (batch['left_iris'] * scale).to(self.device)
            right_iris = (batch['right_iris'] * scale).to(self.device)
            mask_left_iris = (batch['mask_left_iris'] * scale).to(self.device)
            mask_right_iris = (batch['mask_right_iris'] * scale).to(self.device)

            self.diff_renderer.rasterizer.reset()

            best_loss = np.inf
            
            for p in range(iters):
                if p % self.config.raster_update == 0:
                    self.diff_renderer.rasterizer.reset()
                losses = {}

                K, R, T = self.dataset.KRT
                new_K = K.clone()
                new_K[:, :2, :] = K[:, :2, :] * scale
                                
                self.cameras = cameras_from_opencv_projection(R, T, new_K, torch.from_numpy(image_size)[None]).to(self.device)
                vertices, lmk68, lmkMP = self.flame(
                    shape_params=shape,
                    rot_params=matrix_to_rotation_6d(rotation),
                    trans_params=translation,
                    expression_params=expr,
                    eye_pose_params=eyes,
                    jaw_pose_params=jaw,
                    eyelid_params=eyelids
                )

                # reperition for all cameras
                vertices = self.repeat_n_times(vertices, self.num_cameras)
                lmk68 = self.repeat_n_times(lmk68, self.num_cameras)
                lmkMP = self.repeat_n_times(lmkMP, self.num_cameras)

                proj_lmksMP = self.cameras.transform_points_screen(lmkMP)[..., :2]
                proj_lmks68 = self.cameras.transform_points_screen(lmk68)[..., :2]
                proj_vertices = self.cameras.transform_points_screen(vertices)[..., :2]

                right_eye, left_eye = eyes[:, :6], eyes[:, 6:]
                
                lmk_mask = lmk_mask.to(self.device)
                lmk_dense_mask = lmk_dense_mask.to(self.device)

                # Landmarks sparse term
                losses['loss/lmk_oval'] = util.oval_lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_oval
                losses['loss/lmk_68'] = util.lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_68
                losses['loss/lmk_MP'] = util.face_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks
                losses['loss/lmk_eye'] = util.eye_closure_lmk_loss(proj_lmksMP, image_lmksMP, image_size, lmk_dense_mask) * self.config.w_lmks_lid
                losses['loss/lmk_mouth'] = util.mouth_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks_mouth
                losses['loss/lmk_iris_left'] = util.lmk_loss(proj_vertices[:, left_iris_flame, ...], left_iris, image_size, mask_left_iris) * self.config.w_lmks_iris
                losses['loss/lmk_iris_right'] = util.lmk_loss(proj_vertices[:, right_iris_flame, ...], right_iris, image_size, mask_right_iris) * self.config.w_lmks_iris

                # Reguralizers
                losses['reg/expr'] = torch.sum(expr ** 2) * self.config.w_exp
                losses['reg/sym'] = torch.sum((right_eye - left_eye) ** 2) * 8.0
                losses['reg/jaw'] = torch.sum((I6D - jaw) ** 2) * self.config.w_jaw
                losses['reg/eye_lids'] = torch.sum((eyelids[:, 0] - eyelids[:, 1]) ** 2)
                losses['reg/eye_left'] = torch.sum((I6D - left_eye) ** 2)
                losses['reg/eye_right'] = torch.sum((I6D - right_eye) ** 2)
                losses['reg/shape'] = torch.sum(shape ** 2) * self.config.w_shape
                losses['reg/tex'] = torch.sum(tex ** 2) * self.config.w_tex
                losses['reg/pp'] = torch.sum(pp ** 2)
                
                
                # Dense term (look at the config pyr_levels)
                if k > 0 or self.is_initializing:
                    albedos = self.flametex(tex)
                    ops = self.diff_renderer(vertices, albedos, sh, self.cameras)

                    # Photometric dense term
                    grid = ops['position_images'].permute(0, 2, 3, 1)[:, :, :, :2]
                    flipped = flipped.to(self.device)
                    sampled_image = F.grid_sample(flipped, grid * aspect_ratio, align_corners=False)

                    losses['loss/pho'] = util.pixel_loss(ops['images'], sampled_image, self.parse_mask(ops, batch)) * pho_weight_func(k)
                    # pass

                all_loss = self.reduce_loss(losses)
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                for key in losses.keys():
                    self.writer.add_scalar(key, losses[key], global_step=self.global_step)

                self.global_step += 1

                if p % iters == 0:
                    logs.append(f"Color loss for level {k} [frame {str(self.frame).zfill(4)}] =" + reduce(lambda a, b: a + f' {b}={round(losses[b].item(), 4)}', [""] + list(losses.keys())))

                loss_color = all_loss.item()

                if loss_color < best_loss:
                    best_loss = loss_color
                    self.update(optimizer.param_groups)

        for log in logs: logger.info(log)

    def get_mask_tilted_line(self, verts, h=1920, w=1080): 
        # Adjusting input to handle batched input: assume verts_ndc has shape (B, V, 3)
        batch_size = verts.shape[0]
        
        # Scaling verts_xy to image space for each batch
        # verts_xy = (verts_ndc * 0.5 + 0.5) * torch.tensor([w, h], device=verts_ndc.device)
        verts_xy = verts[..., :2]  # Shape: (B, V, 2)
        # Get specific regions for each batch
        verts_xy_left = verts_xy[:, 3193]
        verts_xy_right = verts_xy[:, 3296]
        verts_xy_bottom = verts_xy[:, 3285]
 
        # Compute slope and intercept (k and b) for each item in the batch
        delta_xy = verts_xy_left - verts_xy_right
        assert (delta_xy[:, 0] != 0).all()
        k = delta_xy[:, 1] / delta_xy[:, 0]
        b = verts_xy_bottom[:, 1] - k * verts_xy_bottom[:, 0]

        # Prepare mesh grid for each batch
        x = torch.arange(w, device=verts.device)
        y = torch.arange(h, device=verts.device)
        yx = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)  # Shape: (h, w, 2)
        
        # Expanding yx to match batch dimensions
        yx = yx[None, :, :, :]  # Shape: (1, h, w, 2)

        # Calculate mask for each batch item
        k = k[:, None, None]  # Expand k to (B, 1, 1)
        b = b[:, None, None]  # Expand b to (B, 1, 1)
        mask = ((k * yx[..., 1] + b - yx[..., 0]) > 0).float()  # Shape: (B, h, w)

        # Apply anti-aliasing with Gaussian kernel
        kernel_size = int(0.03 * w) // 2 * 2 + 1
        blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=kernel_size)
        
        # Apply blur for each mask in the batch
        mask = torch.stack([blur(mask[i:i+1])[0] for i in range(batch_size)], dim=0)  # Shape: (B, h, w)

        return mask

    def checkpoint(self, batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.HEATMAP], [View.COLOR_OVERLAY, View.SHAPE_OVERLAY, View.SHAPE, View.SHAPE_MERGE]], frame_dst='/video', save=True, dump_directly=False):
        batch = self.to_cuda(batch)
        images, landmarks, landmarks_dense, _, _ = self.parse_batch(batch)

        # input_image = util.to_image(batch['image'].clone()[0].cpu().numpy())

        savefolder = self.save_folder + self.actor_name + frame_dst
        Path(savefolder).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            # Pytorch3d cameras
            self.cameras = batch['cameras'].to(self.device)
            self.diff_renderer.rasterizer.reset()
            self.diff_renderer.set_size(self.get_image_size())
            self.debug_renderer.rasterizer.raster_settings.image_size = self.get_image_size()

            vertices, lmk68, lmkMP = self.flame(
                shape_params=self.shape,
                rot_params=matrix_to_rotation_6d(self.rotation),
                trans_params=self.translation,
                expression_params=self.expr,
                eye_pose_params=self.eyes,
                jaw_pose_params=self.jaw,
                eyelid_params=self.eyelids
            )
            
            lmk68_n = self.repeat_n_times(lmk68, self.num_cameras)
            lmkMP_n = self.repeat_n_times(lmkMP, self.num_cameras)
            vertices_n = self.repeat_n_times(vertices, self.num_cameras)

            lmk68 = self.cameras.transform_points_screen(lmk68_n, image_size=self.image_size)
            lmkMP = self.cameras.transform_points_screen(lmkMP_n, image_size=self.image_size)
            verts_ndc = self.cameras.transform_points_screen(vertices_n, image_size=self.image_size)

            albedos = self.flametex(self.tex)
            albedos = F.interpolate(albedos, self.get_image_size(), mode='bilinear')
            ops = self.diff_renderer(vertices_n, albedos, self.sh, cameras=self.cameras)
            mask = (self.parse_mask(ops, batch, visualization=True) > 0).float()
            predicted_images = (ops['images'] * mask + (images * (1.0 - mask)))
            shape_mask = ((ops['alpha_images'] * ops['mask_images_mesh']) > 0.).int()

            final_views = []

            for views in visualizations:
                row = []
                for view in views:
                    if view == View.COLOR_OVERLAY:
                        row.append(predicted_images.cpu().numpy())
                    if view == View.GROUND_TRUTH:
                        row.append(images.cpu().numpy())
                    if view == View.SHAPE:
                        shape = self.render_shape(vertices_n, white=False).cpu().numpy()
                        row.append(shape)
                    if view == View.SHAPE_MERGE:
                        shape = self.render_shape(vertices_n, white=False)
                        background_tensor = torch.tensor([255, 255, 255]).reshape(1, 3, 1, 1).expand_as(images).byte().to(self.device)
                        blend = images.cpu().numpy() * 0.5 + shape.cpu().numpy() * 0.5
                        row.append(blend)
                    if view == View.LANDMARKS:
                        gt_lmks = images.clone()
                        gt_lmks = util.tensor_vis_landmarks(gt_lmks, torch.cat([landmarks_dense, landmarks[:, :17, :]], dim=1), color='g')
                        gt_lmks = util.tensor_vis_landmarks(gt_lmks, torch.cat([lmkMP, lmk68[:, :17, :]], dim=1), color='r')
                        row.append(gt_lmks.cpu().numpy())
                    if view == View.SHAPE_OVERLAY:
                        shape = self.render_shape(vertices_n, white=False) * shape_mask
                        blend = images * (1 - shape_mask) + images * shape_mask * 0.3 + shape * 0.7 * shape_mask
                        row.append(blend.cpu().numpy())
                    if view == View.HEATMAP:
                        t = images[0].cpu()
                        f = predicted_images.cpu()
                        l2 = torch.pow(torch.abs(f - t), 2)
                        heatmap = self.get_heatmap(l2[None])
                        row.append(heatmap)
                final_views.append(row)

            # VIDEO
            final_views = util.merge_views(final_views)
            concatenated_views = np.concatenate(final_views, axis=1)
            frame_id = str(self.frame).zfill(5)

            cv2.imwrite('{}/{}.jpg'.format(savefolder, frame_id), concatenated_views)
            # cv2.imwrite('{}/{}.png'.format(self.input_folder, frame_id), input_image)

            if not save:
                return

            # CHECKPOINT
            self.save_checkpoint(frame_id)

            # DEPTH
            # depth_view = self.diff_renderer.render_depth(vertices_n, cameras=self.cameras, faces=torch.cat([util.get_flame_extra_faces(), self.diff_renderer.faces], dim=1))
            # depth = depth_view[0].permute(1, 2, 0)[..., 2:].cpu().numpy() * 1000.0
            # cv2.imwrite('{}/{}.png'.format(self.depth_folder, frame_id), depth.astype(np.uint16))

            # mask = self.get_mask_tilted_line(verts_ndc)
            # transform = T.ToPILImage()
            # for j in range(self.num_cameras):
            #     rgb_path = self.config.flame_param_path + "/" + f"{self.frame:05d}_{j:02d}.png"
            #     mask_path = rgb_path.replace("images", "fg_masks")
            #     image = F_t.to_tensor(Image.open(rgb_path).convert("RGBA")).to(self.device)
            #     fg_mask = F_t.to_tensor(Image.open(mask_path).convert("RGB")).to(self.device)

            #     # Create a white background (1, H, W) - in this case, white is [1, 1, 1] for RGB
            #     white_background = torch.ones_like(image[:3, :, :], device=self.device)  # RGB channels only
            #     # Set the alpha channel
            #     alpha = image[3, :, :]  # Alpha channel (transparency)
            #     # Replace areas where alpha is 0 (transparent) with the white background
            #     image[:3, :, :] = alpha.unsqueeze(0) * image[:3, :, :] + (1 - alpha.unsqueeze(0)) * white_background
            #     # Now image is RGB, with a white background in transparent regions
            #     image = image[:3, :, :]  # Drop alpha channel, keeping only RGB

            #     mask_float = mask[j].float()
            #     image = image * mask_float[None, :, :] + (1 - mask_float[None, :, :]) * white_background
            #     fg_mask = mask[j] * fg_mask 
            #     img_save_path = f"{self.save_folder}" + self.actor_name + "/images/" + f"{self.frame:05d}_{j:02d}.png"
            #     mask_save_path = img_save_path.replace("images", "fg_masks")
            #     if not os.path.exists(os.path.dirname(img_save_path)):
            #         os.makedirs(os.path.dirname(img_save_path))
            #         os.makedirs(os.path.dirname(mask_save_path))
            #     img = transform(image.cpu())
            #     fg_mask = transform(fg_mask.cpu())
            #     img.save(img_save_path)
            #     fg_mask.save(mask_save_path)

    def optimize_frame(self, batch):
        batch = self.to_cuda(batch)
        images = self.parse_batch(batch)[0]
        h, w = images.shape[2:4]
        pyramid_size = np.array([h, w])
        pyramid = util.get_gaussian_pyramid([(pyramid_size * size, util.round_up_to_odd(steps)) for size, steps in self.pyr_levels], images, self.kernel_size, self.sigma)
        self.optimize_color(batch, pyramid, self.clone_params_tracking, lambda k: self.config.w_pho, reg_from_prev=True)
        self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.COLOR_OVERLAY, View.LANDMARKS, View.SHAPE_OVERLAY]])

    def optimize_video(self):
        self.is_initializing = False
        for i in list(range(self.frame, len(self.dataset))):
            batch = self.dataset[i]
            if type(batch) is torch.Tensor:
                continue
            self.optimize_frame(batch)
            self.frame += 1

    def output_video(self):
        util.images_to_video(self.output_folder, self.config.fps)

    def parse_batch(self, batch):
        images = batch['image']
        landmarks = batch['lmk']
        landmarks_dense = batch['dense_lmk']

        lmk_dense_mask = ~(landmarks_dense.sum(2, keepdim=True) == 0)
        lmk_mask = ~(landmarks.sum(2, keepdim=True) == 0)

        left_iris = landmarks_dense[:, left_iris_mp, :]
        right_iris = landmarks_dense[:, right_iris_mp, :]
        mask_left_iris = lmk_dense_mask[:, left_iris_mp, :]
        mask_right_iris = lmk_dense_mask[:, right_iris_mp, :]

        batch['left_iris'] = left_iris
        batch['right_iris'] = right_iris
        batch['mask_left_iris'] = mask_left_iris
        batch['mask_right_iris'] = mask_right_iris

        return images, landmarks, landmarks_dense[:, mediapipe_idx, :2], lmk_dense_mask[:, mediapipe_idx, :], lmk_mask

    def prepare_data(self):
        self.dataset = VideoDataset(self.config, img_to_tensor=True, batchify_all_views=True)
        if not self.dataset.get_property_path("kpt", -1).parent.exists():
            self.dataset.detect_dataset()
        # self.dataset = ImagesDataset(self.config)
        # 
        # self.dataloader = DataLoader(self.dataset, batch_size=None, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        self.num_cameras = self.dataset.num_cameras
        self.num_timesteps = self.dataset.num_timesteps

    def initialize_tracking(self):
        self.is_initializing = True
        keyframes = self.config.keyframes
        if len(keyframes) == 0:
            logger.error('[ERROR] Keyframes are empty!')
            exit(0)
        keyframes.insert(0, keyframes[0])
        for i, j in enumerate(keyframes):
            # batch = self.to_cuda(self.dataset[j], unsqueeze=True)
            batch = self.dataset[j]
            images = self.parse_batch(batch)[0]
            h, w = images.shape[2:4]
            pyramid_size = np.array([h, w])
            pyramid = util.get_gaussian_pyramid([(pyramid_size * size, util.round_up_to_odd(steps * 2)) for size, steps in self.pyr_levels], images, self.kernel_size, self.sigma)
            params = self.clone_params_initialization
            if i == 0:
                params = self.clone_params_color
                self.optimize_camera(batch)
                for k, level in enumerate(pyramid):
                    self.save_tensor(level[0], f"{self.pyramid_folder}/{k}.png")
            self.optimize_color(batch, pyramid, params, lambda k: self.config.w_pho)
            self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE_MERGE, View.SHAPE]], frame_dst='/initialization')
            self.frame += 1

        self.save_canonical()

    def run(self):
        self.prepare_data()
        if self.config.flame_param:
            self.dataset_export(self)
            return
        if not self.load_checkpoint():
            self.initialize_tracking()
            self.frame = 0

        self.optimize_video()
        self.output_video()
            
if __name__ == '__main__':
    config = parse_args()
    ff = Tracker(config, device='cuda:0')
    ff.run()