# Imports
from PIL import Image
import torch
import torchvision.transforms.functional as F_t
import torchvision.transforms as T

import os
from torchvision import transforms
from .models.birefnet import BiRefNet
from .utils import check_state_dict
from tqdm import tqdm
import argparse
import numpy as np

transform = T.ToPILImage()

upperbody_parts = {1,2,3,5,6,10,14,15,19,21,22,23,24,25,26,27}

def background_matting(folder_path, output_folder, segmentation_folder):
    # Load Model
    birefnet = BiRefNet(bb_pretrained=False)
    state_dict = torch.load('./BiRefNet-general-epoch_244.pth', map_location='cpu')
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)

    # Load Model
    device = 'cuda'
    torch.set_float32_matmul_precision(['high', 'highest'][0])

    birefnet.to(device)
    birefnet.eval()
    print('BiRefNet is ready to use.')

    # Input Data
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create output directory if it doesn't exist
    images_out = os.path.join(output_folder, 'images')
    masks_out = os.path.join(output_folder, 'fg_masks')
    if not os.path.exists(images_out):
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)

    # Walk through all subfolders and files
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_files.append(os.path.join(root, file))

    # Loop through all image files with a progress bar
    for image_file in tqdm(sorted(image_files), desc="Processing images", unit="image"):
        image = Image.open(image_file)
        seg_file = os.path.join(segmentation_folder, os.path.basename(image_file).replace('.jpg', '_seg.npy'))
        seg_mask = np.load(seg_file,  allow_pickle=True)

        upper_body_mask = torch.from_numpy(np.isin(seg_mask, list(upperbody_parts)).astype(np.uint8)).to(device)
        upper_body_mask = upper_body_mask.unsqueeze(0).repeat(3, 1, 1)

        # Process and predict
        input_images = transform_image(image).unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        
        # Convert predictions to PIL image
        pred_pil = transforms.ToPILImage()(pred)
        
        # Scale proportionally with max length to 1024
        scale_ratio = 1024 / max(image.size)
        scaled_size = (int(image.size[0]), int(image.size[1]))
        # Prepare the image with alpha channel
        image_masked = image.resize((1024, 1024))
        image_masked.putalpha(pred_pil)

        image_file = os.path.splitext(os.path.basename(image_file))[0]
        parts = image_file.split('_')
        parts[0] = str(int(parts[0]) - 1)
        image_file = parts[1] + '_' + parts[0].zfill(2) + '.png'
        
        output_path_mask = os.path.join(images_out, image_file)
        output_path_pred = os.path.join(masks_out, image_file)
        
        image_masked = F_t.to_tensor(image_masked.resize(scaled_size)).to(device)
        white_background = torch.ones_like(image_masked[:3, :, :], device=device)  # RGB channels only

        # alpha = image_masked[3, :, :]  # Alpha channel (transparency)
        # # Replace areas where alpha is 0 (transparent) with the white background
        # image_masked[:3, :, :] = alpha.unsqueeze(0) * image_masked[:3, :, :] + (1 - alpha.unsqueeze(0)) * white_background
        # # Now image is RGB, with a white background in transparent regions
        # image_masked = image_masked[:3, :, :]  # Drop alpha channel, keeping only RGB
        output = torch.where(upper_body_mask == 1, image_masked[:3, :, :], white_background)
        image_masked = transform(output.cpu())
        image_masked.save(output_path_mask)
        pred_pil = pred_pil.resize(scaled_size)
        pred_pil = F_t.to_tensor(pred_pil).to(device)
        pred_pil = torch.where(upper_body_mask == 1, pred_pil, torch.zeros_like(pred_pil))
        pred_pil = transform(pred_pil.cpu())
        pred_pil.save(output_path_pred, format="PNG")

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiRefNet Image Processing")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing input images")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    parser.add_argument("--seg_folder", type=str, help="Path to the segmentation folder")
    args = parser.parse_args()

    background_matting(args.folder_path, args.output_folder)
