#!/bin/bash
#SBATCH --job-name="global_surrogate"
#SBATCH --partition=gpu-v100
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=600 # default in minutes
#SBATCH --account=education-eemcs-msc-am
#SBATCH --ntasks=1


module load 2023r1 python openmpi cuda/11.7 py-pip py-tqdm
python -m pip install --user vtk torch==2.0.1 torchvision

srun python data_driven.py /scratch/boanalikwu/Results/ 20000 MSE UNet $SLURM_JOB_ID

