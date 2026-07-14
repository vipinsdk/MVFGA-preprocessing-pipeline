#!/bin/bash
#SBATCH -p cpu20
#SBATCH --mem=512G
#SBATCH -o ./output/cpu-%j.out
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task 32
 
eval "$(conda shell.bash hook)"
  
conda activate egoevent

python main.py $ARGS
