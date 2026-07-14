#!/bin/bash

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate easymocap

apt-get install libosmesa6-dev -y
pip uninstall pyopengl -y && pip install pyopengl==3.1.5
# conda install conda-forge::fvcore pytorch3d::pytorch3d -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

export PYOPENGL_PLATFORM=osmesa