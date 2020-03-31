#!/bin/bash
#SBATCH -c 24
#SBATCH --mem 48G
#SBATCH -t 10:00:00

srun python3 param_fitting.py
