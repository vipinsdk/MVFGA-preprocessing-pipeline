
import torch
import numpy as np
import json
from pathlib import Path

from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

class Pytorch3DRenderer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0")
        self.image_size = torch.tensor([[self.config['image_size'][0], self.config['image_size'][1]]]).to(self.device)
        self.load_camera_params()
        self.setup_renderer()

    def get_image_size(self):
        return self.config['image_size']
    
    @property
    def num_cameras(self):
        return len(self.camera_ids)
    
    def repeat_n_times(self, x: torch.Tensor, n: int):
        """Expand a tensor from shape [F, ...] to [F*n, ...]"""
        return x.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, *x.shape[1:])

    def load_camera_params(self):
        load_path = Path(self.config['root_folder'], 'camera_params.json')
        assert load_path.exists()
        param = json.load(open(load_path))

        self.camera_ids =  list(param["intrinsics"].keys())
        K = torch.tensor([param["intrinsics"][k] for k in self.camera_ids])
        extrinsic = torch.tensor([param["world_2_cam"][k] for k in self.camera_ids])  # (N, 4, 4)
        dist = torch.tensor([param["dist"][k] for k in self.camera_ids])

        self.camera_params = {}
        for i, camera_id in enumerate(self.camera_ids):
            self.camera_params[camera_id] = {"intrinsic": K[i], "extrinsic": extrinsic[i], "dist": dist[i]}
        
        self.R = torch.stack([self.camera_params[camera_id]['extrinsic'][..., :3, :3] for camera_id in self.camera_ids]).to(self.device)
        self.T = torch.stack([self.camera_params[camera_id]['extrinsic'][..., :3, 3] for camera_id in self.camera_ids]).to(self.device)
        self.K = K.to(self.device)
        
        # camera parameters
        self.image_size = torch.tensor([[self.config['image_size'][0], self.config['image_size'][1]]]).cuda()
        self.cameras = cameras_from_opencv_projection(self.R, self.T, self.K, self.image_size).to(self.device)

    def setup_renderer(self):
        mesh_file = '/netscratch/jeetmal/models/metrical-tracker/data/smplx_uv.obj'
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

    def render_shape(self, vertices, images, faces=None, white=True):
        vertices = self.repeat_n_times(torch.from_numpy(vertices).to(self.device), self.num_cameras)
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
        result = rendering[:, :3, :, :]

        result = np.concatenate(result.cpu().numpy(), axis=2)
        result = (result.transpose(1, 2, 0) * 255).astype(np.uint8)
        result = np.minimum(np.maximum(result, 0), 255).astype(np.uint8)

        images = np.concatenate(images, axis=1)
        result = images * 0.4 + result * 0.6
        return result
        # return rendering[:, 0:3, :, :]
    


