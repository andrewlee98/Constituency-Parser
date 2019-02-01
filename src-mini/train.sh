#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=TrainParser # The job name.
#SBATCH -c 1 # The number of cpu cores to use.
#SBATCH --time=1:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=1gb # The memory the job will use per cpu core.
 
module load anaconda
 
#Command to execute Python program
source ../../test_py3/bin/activate
python3 train.py
 
#End of script
