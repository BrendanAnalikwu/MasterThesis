#!/bin/bash
#SBATCH --job-name="best bo result"
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=1440 # default in minutes
#SBATCH --account=education-eemcs-msc-am
#SBATCH --ntasks=1
#SBATCH --array=1-8
#SBATCH --signal=SIGTERM@300


module load 2023r1 python py-numpy/1.22.4 openmpi cuda/11.7 py-pip py-tqdm py-scipy py-scikit-learn
python -m pip install --user vtk torch==2.0.1 torchvision

mkdir -p /tmp/$SLURM_JOB_ID/
start=$SECONDS
unzip -q /scratch/boanalikwu/Results/physical1_tensors.zip -d /tmp/$SLURM_JOB_ID/
echo Unzip took $((SECONDS-start)) seconds

hyperparameters="0 4 -0.5 -1.3 3.0 -2.0 -3.0 -2.684257561047547 -4.0 -1.0"
echo $hyperparameters

srun python bayesian_train.py /tmp/$SLURM_JOB_ID/scratch/boanalikwu/Results/ $SLURM_JOB_ID $(echo $hyperparameters)

rm -r /tmp/$SLURM_JOB_ID/