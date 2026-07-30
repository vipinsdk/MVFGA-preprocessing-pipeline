<div align="center">

# MVFGA Dataset Preprocessing Pipeline

**An end-to-end preprocessing toolkit for multi-view face and gesture animation datasets.**

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge)](https://dfki-av.github.io/MVFGA/)
[![Paper](https://img.shields.io/badge/Paper-Computer%20Graphics%20Forum-4c72b0?style=for-the-badge)](https://diglib.eg.org/server/api/core/bitstreams/e49bfeeb-e6f3-4391-ba4f-52e09bc18386/content)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## Overview

This repository provides the preprocessing pipeline used for **MVFGA (Multi-View Face and Gesture Animation with Dynamic Gaussians)**. It contains scripts for preparing synchronized multi-view recordings, including video trimming, camera calibration, background matting, body segmentation, and landmark detection.

## Pipeline

![MVFGA Preprocessing Pipeline](pipeline.png)

The preprocessing workflow consists of the following stages:

| Stage | Task | Description |
|---:|---|---|
| 1 | Video synchronization and trimming | Aligns multi-camera recordings and trims them to a common temporal range. |
| 2 | Camera calibration | Estimates intrinsic and extrinsic camera parameters from checkerboard recordings. |
| 3 | Background matting | Extracts foreground subjects using BiRefNet. |
| 4 | Body segmentation | Produces human-body segmentation masks using Sapiens. |
| 5 | Landmark detection | Extracts MediaPipe whole-body 2D landmarks through the EasyMoCap wrapper. |

## Installation

Create and activate the Conda environment provided with the repository:

```bash
chmod +x scripts/install_conda.sh
./scripts/install_conda.sh

sudo apt-get update
sudo apt-get install -y git

conda env create -f environment.yml
conda activate mvfga
export CUDA_HOME="$CONDA_PREFIX"
```

Install PyTorch3D:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Install HaMeR and ViTPose:

```bash
cd hamer
pip install -e ".[all]"
pip install -v -e third-party/ViTPose
cd ..
```

Install EasyMoCap in development mode:

```bash
cd Easymocap
python setup.py develop
cd ..
```

Review [`environment.yml`](environment.yml) for the complete list of dependencies and version requirements.

## Usage

### Synchronize multi-view videos

The synchronization interface lets you align all camera streams to the same start or end time. You can move through frames for all videos simultaneously or adjust an individual stream through the GUI controls.

```bash
cd sync_streams
python3 multiview_sync.py --video_dir "D:/Datacapture/Extrinsics_1/videos"
```

### Run the preprocessing pipeline

```bash
python main.py \
  --root_dir /home/vippin/thesis/extra \
  --output /home/vippin/thesis/extra/demo_mvfga \
  --sapiens \
  --calibrate \
  --background_matting \
  --annots
```

The example above enables camera calibration, background matting, Sapiens body segmentation, and landmark annotation generation.

## Processing Stages

### 1. Trim videos

The trimming stage processes synchronized video streams and crops each recording to its selected start and end frames.

### 2. Camera calibration

The calibration stage estimates intrinsic and extrinsic camera parameters using checkerboard recordings.

For additional details, see the [EasyMoCap calibration documentation](Easymocap/README.md).

### 3. Background matting

The background-matting stage separates the subject from the background using **BiRefNet**.

For additional details, see the [BiRefNet documentation](BiRefNet/README.md).

### 4. Body segmentation

The body-segmentation stage generates human segmentation masks using **Sapiens**.

For additional details, see the [Sapiens segmentation documentation](sapiens/lite/docs/SEG_README.md).

### 5. MediaPipe landmark detection

EasyMoCap provides a wrapper for extracting MediaPipe whole-body 2D keypoints from the processed sequences.

## Logging

Each pipeline run writes detailed progress information, warnings, and errors to:

```text
output.log
```

## Citation

Please cite our paper when using MVFGA or this preprocessing pipeline in your research:

```bibtex
@article{10.1111:cgf.70567,
  journal   = {Computer Graphics Forum},
  title     = {{Multi-View Face and Gesture Animation with Dynamic Gaussians}},
  author    = {Javanmardi, A. and Jeetmal, V. K. and Millerdurai, C. and Pagani, A. and Stricker, D.},
  year      = {2026},
  publisher = {The Eurographics Association},
  ISSN      = {1467-8659},
  DOI       = {10.1111/cgf.70567}
}
```

### Referenced methods and tools

<details>
<summary><strong>Sapiens</strong></summary>

```bibtex
@article{khirodkar2024sapiens,
  title   = {Sapiens: Foundation for Human Vision Models},
  author  = {Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal = {arXiv preprint arXiv:2408.12569},
  year    = {2024}
}
```

</details>

<details>
<summary><strong>Background Matting — BiRefNet</strong></summary>

```bibtex
@article{zheng2024birefnet,
  title   = {Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author  = {Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal = {CAAI Artificial Intelligence Research},
  volume  = {3},
  pages   = {9150038},
  year    = {2024}
}
```

</details>

<details>
<summary><strong>EasyMoCap</strong></summary>

```bibtex
@misc{easymocap,
  title        = {EasyMoCap - Make human motion capture easier.},
  howpublished = {GitHub},
  year         = {2021},
  url          = {https://github.com/zju3dv/EasyMocap}
}
```

</details>

<details>
<summary><strong>Metrical Face Tracker</strong></summary>

```bibtex
@proceedings{MICA:ECCV2022,
  author  = {Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
  title   = {Towards Metrical Reconstruction of Human Faces},
  journal = {European Conference on Computer Vision},
  year    = {2022}
}
```

</details>

<details>
<summary><strong>HaMeR</strong></summary>

```bibtex
@inproceedings{pavlakos2024reconstructing,
  title     = {Reconstructing Hands in 3{D} with Transformers},
  author    = {Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
  booktitle = {CVPR},
  year      = {2024}
}
```

</details>

## Acknowledgements

This work was partially funded by the **Horizon Europe** programme under the **IRIS-XR** project, Grant Agreement No. **101298672**.

## Contributing

Contributions are welcome. Please fork the repository, create a focused branch, and submit a pull request with a clear description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).
