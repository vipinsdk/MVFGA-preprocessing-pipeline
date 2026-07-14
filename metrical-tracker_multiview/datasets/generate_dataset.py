import os
from abc import ABC
from glob import glob
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset, default_collate
import torchvision.transforms.functional as F

from pytorch3d.utils import cameras_from_opencv_projection
import PIL.Image as Image

from tqdm import tqdm
from typing import Optional
import torchvision
import json
import math
from face_detector import FaceDetector
from image import crop_image_bbox, squarefiy, get_bbox
from configs.config import parse_args

class GeneratorDataset(Dataset, ABC):
    def __init__(self, source, config):
        self.device = 'cuda:0'
        self.config = config
        self.source = Path(source)
        self.tgt_folder = Path(config.save_folder)
        self.camera_id = str(config.camera_id)

        self.initialize()
        self.load_camera_params()
        
        print(f'Camera Distortion: {self.config.undistort}')
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)

    def initialize(self):
        path = Path(self.source, 'source')
        if not path.exists() or len(os.listdir(str(path))) == 0:
            video_file = self.source / 'video.mp4'
            if not os.path.exists(video_file):
                logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {self.source}')
                exit(1)
            path.mkdir(parents=True, exist_ok=True)
            os.system(f'ffmpeg -i {video_file} -vf fps={self.config.fps} -q:v 1 {self.source}/source/%05d.png')

        self.images = sorted(glob(f'{self.source}/source/*.jpg') + glob(f'{self.source}/source/*.png'))

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks
    
    def load_camera_params(self):
        from pathlib import Path
        load_path = Path(self.config.root_folder, 'camera_params.json')
        assert load_path.exists()
        param = json.load(open(load_path))
        
        self.K = np.array(param['intrinsics'][self.camera_id])
        self.dist = np.array(param['dist'][self.camera_id])

    def run(self):
        logger.info('Generating dataset...')
        bbox = None
        bbox_path = self.config.actor + "/bbox.pt"

        if os.path.exists(bbox_path):
            bbox = torch.load(bbox_path)

        for imagepath in tqdm(self.images):
            lmk_path = imagepath.replace('source', 'kpt').replace('png', 'npy').replace('jpg', 'npy')
            lmk_path_dense = imagepath.replace('source', 'kpt_dense').replace('png', 'npy').replace('jpg', 'npy')

            if not os.path.exists(lmk_path) or not os.path.exists(lmk_path_dense):
                image_dist = cv2.imread(imagepath)
                if self.config.undistort:
                    image = cv2.undistort(image_dist, self.K, self.dist)
                else:
                    image = image_dist
                h, w, c = image.shape

                if bbox is None and self.config.crop_image:
                    lmk, _ = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # estimate initial bbox
                    bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
                    torch.save(bbox, bbox_path)

                if self.config.crop_image:
                    image = crop_image_bbox(image, bbox)
                    if self.config.image_size[0] == self.config.image_size[1]:
                        image = squarefiy(image, size=self.config.image_size[0])
                # else:
                    # image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]), interpolation=cv2.INTER_CUBIC)

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + imagepath)
                    lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + imagepath)
                    dense_lmk = np.zeros([478, 2])

                Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)
                Path(lmk_path_dense).parent.mkdir(parents=True, exist_ok=True)
                Path(imagepath.replace('source', 'images')).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(imagepath.replace('source', 'images'), image)
                np.save(lmk_path_dense, dense_lmk)
                np.save(lmk_path, lmk)


class VideoDataset(Dataset):
    def __init__(
        self,
        config,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |---images/
                |   |---<timestep_id>.jpg
                |
                |---alpha_maps/
                |   |---<timestep_id>.png
                |
                |---landmark2d/
                        |---face-alignment/
                        |    |---<camera_id>.npz
                        |
                        |---MP/
                                |---<camera_id>.npz
        """
        super().__init__()
        self.config = config
        self.device = 'cuda:0'
        self.root_folder = Path(config.root_folder)
        self.tgt_folder = Path(config.save_folder)
        self.img_to_tensor = img_to_tensor
        self.batchify_all_views = batchify_all_views
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)

        self.define_properties()
        self.load_camera_params()

        # timesteps
        image_folder = self.root_folder / self.properties['image']['folder']
        all_timesteps = set(
            f.split('.')[0].split('_')[-1]
            for f in os.listdir(image_folder) if f.endswith(self.properties['image']['suffix'])
        )
        self.timestep_ids = set()
        for timestep in all_timesteps:
            if all((image_folder / f"{camera_id}_{timestep}.{self.properties['image']['suffix']}").exists() for camera_id in self.camera_ids):
                self.timestep_ids.add(timestep)

        self.timestep_ids = sorted(self.timestep_ids)
        self.timestep_indices = list(range(len(self.timestep_ids)))

        logger.info(f"number of timesteps: {self.num_timesteps}, number of cameras: {self.num_cameras}")

        # collect
        self.items = []
        for fi, timestep_index in enumerate(self.timestep_indices):
            for ci, camera_id in enumerate(self.camera_ids):
                self.items.append(
                    {
                        "timestep_index": fi,  # new index after filtering
                        "timestep_index_original": timestep_index,  # original index
                        "timestep_id": self.timestep_ids[timestep_index],
                        "camera_index": ci,
                        "camera_id": camera_id,
                    }
                )
  
    def define_properties(self):
        self.properties = {
            "image": {
                "folder": "source",
                "per_timestep": True,
                "suffix": "jpg",
            },
             "alpha_map": {
                "folder": "alpha_maps",
                "per_timestep": True,
                "suffix": "jpg",
            },
            "kpt": {
                "folder": "kpt",
                "per_timestep": True,
                "suffix": "npy",
            },
            "kpt_dense": {
                "folder": "kpt_dense",
                "per_timestep": True,
                "suffix": "npy",
            },
            "shape": {
                "folder": "shape",
                "per_timestep": False,
                "suffix": "npy",
        }}

    @staticmethod
    def get_number_after_prefix(string, prefix):
        i = string.find(prefix)
        if i != -1:
            number_begin = i + len(prefix)
            assert number_begin < len(string), f"No number found behind prefix '{prefix}'"
            assert string[number_begin].isdigit(), f"No number found behind prefix '{prefix}'"

            non_digit_indices = [i for i, c in enumerate(string[number_begin:]) if not c.isdigit()]
            if len(non_digit_indices) > 0:
                number_end = number_begin + min(non_digit_indices)
                return int(string[number_begin:number_end])
            else:
                return int(string[number_begin:])
        else:
            return None
    
    def load_camera_params(self):
        from pathlib import Path
        load_path = Path(self.config.root_folder, 'camera_params.json')
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
        self.image_size = torch.tensor([[self.config.image_size[0], self.config.image_size[1]]]).cuda()
        self.cameras = cameras_from_opencv_projection(self.R, self.T, self.K, self.image_size).to(self.device)
        self.principal_point = torch.Tensor([[(2 * K[0][2] / self.config.image_size[1] - 1, 2 * K[1][2] / self.config.image_size[0] - 1) for K in self.K]]).to(self.device)

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks
    
    def __len__(self):
        if self.batchify_all_views:
            return self.num_timesteps
        else:
            return len(self.items)

    def __getitem__(self, i):
        if self.batchify_all_views:
            return self.getitem_by_timestep(i)
        else:
            return self.getitem_single_image(i)


    def getitem_single_image(self, i):
        item = self.items[i].copy()

        rgb_path = self.get_property_path("image", i)
        item["image"] = Image.open(rgb_path).convert("RGB")

        # shape_path = self.get_property_path("shape", i)
        # item["shape"] = torch.from_numpy(np.load(shape_path)).float()

        camera_param = self.camera_params[item["camera_id"]]
        item["intrinsic"] = camera_param["intrinsic"].clone()
        item["extrinsic"] = camera_param["extrinsic"].clone()
        item["dist"] = camera_param["dist"].clone()

        timestep_index = self.items[i]["timestep_index"]
        camera_index = self.items[i]["camera_index"]
        
        if self.config.alpha_map:
            # alpha_path = self.get_property_path("alpha_map", i)
            alpha_path = self.config.root_folder + "/alpha_maps/" + f"{timestep_index:05d}_{camera_index:02d}.png"
            item["alpha_map"] = np.array(Image.open(alpha_path))

        # undistort the image
        if self.config.undistort:
            image = np.array(item["image"])[:, :, ::-1].copy()
            image = cv2.undistort(image, item["intrinsic"].cpu().numpy(), item["dist"].cpu().numpy())
            item["image"] = Image.fromarray(image[:, :, ::-1])

        landmark_path = self.get_property_path("kpt", i)
        dense_landmark_path = self.get_property_path("kpt_dense", i)

        landmark_npz = torch.from_numpy(np.load(landmark_path)).squeeze(0)
        dense_landmark_npz = torch.from_numpy(np.load(dense_landmark_path)).squeeze(0)

        item["lmk"] = landmark_npz  # (num_points, 3)
        item["dense_lmk"] = dense_landmark_npz  # (num_points, 3)

        item = self.apply_transforms(item)
        return item

    def getitem_by_timestep(self, timestep_index):
        begin = timestep_index * self.num_cameras
        indices = range(begin, begin + self.num_cameras)
        item = default_collate([self.getitem_single_image(i) for i in indices])

        if self.config.flame_param:
            flame_param_path = self.config.flame_param_path + "/" + f"{timestep_index:05d}.npz"
            item["flame_param"] = dict(np.load(flame_param_path))

        item["num_cameras"] = self.num_cameras
        item["cameras"] = self.cameras
        item["principal_point"] = self.principal_point

        return item

    def apply_transforms(self, item):
        # item = self.apply_scale_factor(item)
        item = self.apply_to_tensor(item)
        return item

    def apply_to_tensor(self, item):
        if self.img_to_tensor:
            if "image" in item:
                item["image"] = F.to_tensor(item["image"])
            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])
        return item

    def apply_scale_factor(self, item):
        if "rgb" in item:
            H, W, _ = item["rgb"].shape
            h, w = int(H * self.cfg.scale_factor), int(W * self.cfg.scale_factor)
            rgb = Image.fromarray(item["rgb"]).resize(
                (w, h), resample=Image.BILINEAR
            )
            item["rgb"] = np.array(rgb)
    
        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h
        
        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

        # properties need to be scaled down when rgb is downsampled
        n_downsample_rgb = self.cfg.n_downsample_rgb if self.cfg.n_downsample_rgb else 1
        scale_factor = self.cfg.scale_factor / n_downsample_rgb
        item["scale_factor"] = scale_factor  # NOTE: not self.cfg.scale_factor
        if scale_factor < 1.0:
            if "intrinsic" in item:
                item["intrinsic"][:2] *= scale_factor
            if "alpha_map" in item:
                h, w = item["rgb"].shape[:2]
                alpha_map = Image.fromarray(item["alpha_map"]).resize(
                    (w, h), Image.Resampling.BILINEAR
                )
                item["alpha_map"] = np.array(alpha_map)
        return item

    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        folder = p["folder"] if "folder" in p else None
        per_timestep = p["per_timestep"]
        suffix = p["suffix"]

        path = self.root_folder
        if folder is not None:
            path = path / folder

        if self.num_cameras > 1:
            if camera_id is None:
                assert (
                    index is not None), "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_prefix" in p:
                camera_id = p["cam_id_prefix"] + camera_id
        else:
            camera_id = ""

        if per_timestep:
            if timestep_id is None:
                assert index is not None, "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            if len(camera_id) > 0:
                path /= f"{camera_id}_{timestep_id}.{suffix}"
            else:
                path /= f"{timestep_id}.{suffix}"
        else:
            if len(camera_id) > 0:
                path /= f"{camera_id}.{suffix}"
            else:
                path = Path(str(path) + f".{suffix}")

        return path
        
    def get_property_path_list(self, name):
        paths = []
        for i in range(len(self.items)):
            img_path = self.get_property_path(name, i)
            paths.append(img_path)
        return paths

    @property
    def num_timesteps(self):
        return len(self.timestep_indices)

    @property
    def KRT(self):
        return self.K, self.R, self.T
    
    @property
    def num_cameras(self):
        return len(self.camera_ids)
    
    def detect_dataset(self):
        """
        Annotates each frame with facial landmarks
        :return: dict mapping frame number to landmarks numpy array
        """

        logger.info("Camera distrotion: " + str(self.config.undistort))
        logger.info("Begin annotating landmarks...")

        # for i, item in enumerate(tqdm(self.items)):
        for _ , camera_id in enumerate(self.camera_ids):
            self.face_detector_mediapipe = FaceDetector('google')
            for _ , timestep_index in enumerate(self.timestep_indices):
                rgb_path = str(self.get_property_path("image", camera_id=camera_id, timestep_id=self.timestep_ids[timestep_index]))
                out_path = self.get_property_path("kpt", camera_id=camera_id, timestep_id=self.timestep_ids[timestep_index])
                out_path_dense = self.get_property_path("kpt_dense", camera_id=camera_id, timestep_id=self.timestep_ids[timestep_index])
                
                if not os.path.exists(rgb_path):
                    continue
                
                if os.path.exists(out_path):
                        continue

                logger.info(f"Processing {rgb_path}")

                camera_param = self.camera_params[camera_id]
                K = camera_param["intrinsic"].clone().cpu().numpy()
                dist = camera_param["dist"].clone().cpu().numpy()

                image_dist = cv2.imread(rgb_path)
                if self.config.undistort:
                    image = cv2.undistort(image_dist, K, dist)
                else:
                    image = image_dist

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + str(rgb_path))
                    lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + str(rgb_path))
                    dense_lmk = np.zeros([478, 2])

                if not out_path.parent.exists():
                        out_path.parent.mkdir(parents=True)
                np.save(out_path, lmk)

                if not out_path_dense.parent.exists():
                    out_path_dense.parent.mkdir(parents=True)
                np.save(out_path_dense, dense_lmk)
            

    def annotate_landmarks(self):
        """
        Annotates each frame with landmarks for face and iris. Assumes frames have been extracted
        :return:
        """

        lmks, lmks_dense = self.detect_dataset()

        # construct final json
        for camera_id, lmk_face_camera in lmks.items():
            face_landmark_2d_dense = []
            face_landmark_2d = []
            for timestep_id in lmk_face_camera.keys():
                face_landmark_2d_dense.append(lmks_dense[camera_id][timestep_id][None])
                face_landmark_2d.append(lmks[camera_id][timestep_id][None])

            lmk_dict = {
                "face_landmark_2d": face_landmark_2d,
            }

            lmk_dense_dict = {
                "face_landmark_2d_dense": face_landmark_2d_dense,
            }

            for k, v in lmk_dict.items():
                if len(v) > 0:
                    lmk_dict[k] = np.concatenate(v, axis=0)
            
            for k, v in lmk_dense_dict.items():
                if len(v) > 0:
                     lmk_dense_dict[k] = np.concatenate(v, axis=0)

            # face alignemt landmarks
            out_path = self.get_property_path(
                "kpt", camera_id=camera_id
            )
            logger.info(f"Saving landmarks to: {out_path} of length {len(lmk_dict['face_landmark_2d'])}")
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True)
            np.savez(out_path, **lmk_dict)

            # mediapipe landmarks
            out_path_dense = self.get_property_path(
                "kpt_dense", camera_id=camera_id
            )
            logger.info(f"Saving dense landmarks to: {out_path_dense}")
            if not out_path_dense.parent.exists():
                out_path_dense.parent.mkdir(parents=True)
            np.savez(out_path_dense, **lmk_dense_dict)

    def write(self, dataloader):
        if not self.tgt_folder.exists():
            self.tgt_folder.mkdir(parents=True)
        
        db = {
            "frames": [],
        }
        
        print(f"Writing images to {self.tgt_folder}")
        timestep_indices = set()
        camera_indices = set()
        for i, item in tqdm(enumerate(dataloader), total=len(dataloader)):
            # print(item.keys())

            timestep_indices.add(item['timestep_index'])
            camera_indices.add(item['camera_index'])

            extrinsic = item['extrinsic'].numpy()
            intrinsic = item['intrinsic'].double().numpy()

            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            fl_x = intrinsic[0, 0]
            fl_y = intrinsic[1, 1]
            h = item['image'].shape[1]
            w = item['image'].shape[2]
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            frame_item = {
                "timestep_index": item['timestep_index'],
                "timestep_index_original": item['timestep_index_original'],
                "timestep_id": item['timestep_id'],
                "camera_index": item['camera_index'],
                "camera_id": item['camera_id'],

                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                
                "transform_matrix": extrinsic.tolist(),
                "flame_param_path" : f"flame_param/{item['timestep_index']:05d}.npz",
                "file_path": f"images/{item['timestep_index']:05d}_{item['camera_index']:02d}.png",
            }
            
            path2data = {
                str(self.tgt_folder / frame_item['file_path']): item['image'],
            }

            if 'alpha_map' in item:
                frame_item['fg_mask_path'] = f"fg_masks/{item['timestep_index']:05d}_{item['camera_index']:02d}.png"
                path2data[str(self.tgt_folder / frame_item['fg_mask_path'])] = item['alpha_map']

            db['frames'].append(frame_item)
            # worker_args.append([path2data])
            # write_data(path2data)

        # add indices to ease filtering
        db['timestep_indices'] = sorted(list(timestep_indices))
        db['camera_indices'] = sorted(list(camera_indices))
        
        write_json(db, self.tgt_folder)
        write_json(db, self.tgt_folder, division='backup')

def write_data(path2data):
    from torchvision.utils import save_image
    for path, data in path2data.items():
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            save_image(data, path)
            # Image.fromarray(np.array(data)).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def write_json(db, tgt_folder, division=None):
    fname = "transforms.json" if division is None else f"transforms_{division}.json"
    json_path = tgt_folder / fname
    print(f"Writing database: {json_path}")
    with open(json_path, "w") as f:
        json.dump(db, f, indent=4)

def split_json(tgt_folder: Path, train_ratio=0.7):
    db = json.load(open(tgt_folder / "transforms.json", "r"))
    
    # init db for each division
    db_train = {k: v for k, v in db.items() if k not in ['frames', 'timestep_indices', 'camera_indices']}
    db_train['frames'] = []
    db_val = {'frames' : []}
    db_test = {'frames' : []}
    # divide timesteps
    nt = len(db['timestep_indices'])
    assert 0 < train_ratio <= 1
    nt_train = int(np.ceil(nt * train_ratio))
    nt_test = nt - nt_train

    # record number of timesteps
    timestep_indices = sorted(db['timestep_indices'])
    db_train['timestep_indices'] = timestep_indices[:nt_train]
    db_val['timestep_indices'] = timestep_indices[:nt_train]  # validation set share the same timesteps with training set
    db_test['timestep_indices'] = timestep_indices[nt_train:]

    if len(db['camera_indices']) > 1:
        # when having multiple cameras, leave one camera for validation (novel-view sythesis)
        if 8 in db['camera_indices']:
            # use camera 8 for validation (front-view of the NeRSemble dataset)
            db_train['camera_indices'] = [i for i in db['camera_indices'] if i != 8]
            db_val['camera_indices'] = [8]
            db_test['camera_indices'] = db['camera_indices']
        else:
            # use the last camera for validation
            db_train['camera_indices'] = db['camera_indices'][:-1]
            db_val['camera_indices'] = [db['camera_indices'][-1]]
            db_test['camera_indices'] = db['camera_indices']
    else:
        # when only having one camera, we create an empty validation set
        db_train['camera_indices'] = db['camera_indices']
        db_val['camera_indices'] = []
        db_test['camera_indices'] = db['camera_indices']

    # fill data by timestep index
    range_train = range(db_train['timestep_indices'][0], db_train['timestep_indices'][-1]+1) if nt_train > 0 else []
    range_test = range(db_test['timestep_indices'][0], db_test['timestep_indices'][-1]+1) if nt_test > 0 else []
    for f in db['frames']:
        if f['timestep_index'] in range_train:
            if f['camera_index'] in db_train['camera_indices']:
                db_train['frames'].append(f)
            elif f['camera_index'] in db_val['camera_indices']:
                db_val['frames'].append(f)
            else:
                raise ValueError(f"Unknown camera index: {f['camera_index']}")
        elif f['timestep_index'] in range_test:
            db_test['frames'].append(f)
            assert f['camera_index'] in db_test['camera_indices'], f"Unknown camera index: {f['camera_index']}"
        else:
            raise ValueError(f"Unknown timestep index: {f['timestep_index']}")
    
    write_json(db_train, tgt_folder, division='train')
    write_json(db_val, tgt_folder, division='val')
    write_json(db_test, tgt_folder, division='test')

if __name__ == '__main__':
    config = parse_args()
    dataset = VideoDataset(config, img_to_tensor=True, batchify_all_views=True)
    dataset.write()