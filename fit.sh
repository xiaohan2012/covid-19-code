#!/bin/bash
#SBATCH -c 20
#SBATCH --mem 40G
#SBATCH -t 10:00:00

srun python3 param_fitting.py
