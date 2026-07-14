from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch
import json

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2_list: list,
                 boxes_list: list,
                 right_list: list,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2_list = img_cv2_list  # List of 15 images per hand
        assert len(img_cv2_list) == len(boxes_list) == len(right_list), "Mismatch in input data lengths"
        
        # Ensure this dataset is for inference only
        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        
        # Preprocess annotations for all 15 images per hand
        self.data = []
        for img_cv2, boxes, right in zip(img_cv2_list, boxes_list, right_list):
            boxes = boxes.astype(np.float32)
            center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
            scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
            personid = np.arange(len(boxes), dtype=np.int32)
            right = right.astype(np.float32)

            self.data.append({
                'img_cv2': img_cv2,
                'center': center,
                'scale': scale,
                'personid': personid,
                'right': right
            })

    def __len__(self) -> int:
        return sum(len(entry['personid']) for entry in self.data)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        # Locate the correct image and hand within the dataset
        cumulative = 0
        for entry in self.data:
            num_persons = len(entry['personid'])
            if cumulative + num_persons > idx:
                person_idx = idx - cumulative
                img_cv2 = entry['img_cv2']
                center = entry['center'][person_idx].copy()
                scale = entry['scale'][person_idx]
                personid = entry['personid'][person_idx]
                right = entry['right'][person_idx].copy()
                break
            cumulative += num_persons

        center_x, center_y = center

        # Define bounding box size and aspect ratio
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        flip = right == 0

        # Generate image patch
        cvimg = img_cv2.copy()
        downsampling_factor = ((bbox_size * 1.0) / patch_width)
        downsampling_factor = downsampling_factor / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans = generate_image_patch_cv2(cvimg, center_x, center_y,
                                                       bbox_size, bbox_size,
                                                       patch_width, patch_height,
                                                       flip, 1.0, 0,
                                                       border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # Apply normalization
        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(personid),
        }
        item['box_center'] = center
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = right
        item['trans'] = trans
        return item
