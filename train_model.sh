#!/bin/bash
#SBATCH --job-name="global_surrogate"
#SBATCH --partition=compute
#SBATCH --mem=20G
#SBATCH --time=400 # default in minutes
#SBATCH --account=education-eemcs-msc-am
#SBATCH --ntasks=1


module load 2023r1 python openmpi cuda/11.7 py-pip py-tqdm
python -m pip install --user vtk torch==2.0.1 torchvision

srun python data_driven.py /scratch/boanalikwu/Results/ 16 16 32 0 0 20000 1 $SLURM_ARRAY_TASK_ID

