#!/bin/bash
#SBATCH -c 32
#SBATCH --mem 20G
#SBATCH -t 10:00:00

srun python3 param_fitting.py
