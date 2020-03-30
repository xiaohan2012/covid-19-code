#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20G

srun python3 param_fitting.py
