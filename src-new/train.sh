#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=TrainParser # The job name.
#SBATCH -c 1 # The number of cpu cores to use.
#SBATCH --time=01:30:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=1gb # The memory the job will use per cpu core.
 
module load anaconda
 
#Command to execute Python program
module load anaconda/3-4.4.0
module load cuda80/toolkit cuda80/blas cudnn/5.1
source activate myenv
python3 train.py
 
#End of script
