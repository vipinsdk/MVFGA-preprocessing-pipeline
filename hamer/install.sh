#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
   source /root/miniconda3/etc/profile.d/conda.sh
	 conda activate hamer

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