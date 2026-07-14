#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
   source /root/miniconda3/etc/profile.d/conda.sh
	 conda activate easymocap

   apt-get install libosmesa6-dev -y
   pip uninstall pyopengl -y && pip install pyopengl==3.1.5
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  # conda install conda-forge::fvcore pytorch3d::pytorch3d -y
   
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