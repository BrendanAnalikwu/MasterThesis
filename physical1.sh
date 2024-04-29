#!/bin/bash
#SBATCH --job-name="No transform"
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=200 # default in minutes
#SBATCH --account=education-eemcs-msc-am
#SBATCH --ntasks=1
#SBATCH --array=1-2


module load 2023r1 python openmpi cuda/11.7 py-pip py-tqdm
python -m pip install --user vtk torch==2.0.1 torchvision

mkdir -p /tmp/$SLURM_JOB_ID/
start=$SECONDS
unzip -q /scratch/boanalikwu/Results/physical1.zip -d /tmp/$SLURM_JOB_ID/
echo Unzip took $((SECONDS-start)) seconds

srun python data_driven.py /tmp/$SLURM_JOB_ID/scratch/boanalikwu/Results/ 2000 MSE UNet -j $SLURM_JOB_ID -b .98 .999 -B 16 -p 10 -a 1

rm -r /tmp/$SLURM_JOB_ID/