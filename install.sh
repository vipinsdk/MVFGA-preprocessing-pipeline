#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  source /root/miniconda3/etc/profile.d/conda.sh
	conda activate mvfga

  #pip install git+https://github.com/facebookresearch/pytorch3d.git
  #  pip install trimesh
   apt-get update
   apt-get install -y libegl1-mesa-dev \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    python3-opengl
    apt-get install -y ffmpeg

    # conda install -c conda-forge colmap -y
    # pip install pycolmap

    export PYOPENGL_PLATFORM=osmesa
   
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

export MKL_NUM_THREADS=1 # Intel Math Kernel Library: used by numpy if installed via conda
export OMP_NUM_THREADS=1 # OpenMP: used by numpy if installed via pip
export OPENBLAS_NUM_THREADS=1 # OpenBLAS: used by numpy
export NUMEXPR_NUM_THREADS=1 # controls multithreading in numexpr package
export USE_OPENMP=1 

# This runs your wrapped command
"$@"