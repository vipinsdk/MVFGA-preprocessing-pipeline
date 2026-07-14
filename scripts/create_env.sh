#!/bin/bash 

chmod +x scripts/install_conda.sh && scripts/install_conda.sh &&  source /root/miniconda3/etc/profile.d/conda.sh
apt-get install git -y
conda env create -f environment.yml
conda activate mvfga
export CUDA_HOME=$CONDA_PREFIX

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

cd hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose

cd ..
cd EasyMocap
python setup.py develop

apt-get clean