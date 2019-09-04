#!/bin/bash

#SBATCH -N 1                         # number of computing nodes 
#SBATCH -n 3                        # number of cores
#SBATCH --time=01-00:00:00           # Max time for task. Format is DD-HH:MM:SS
#SBATCH -o slurm.spz.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.spz.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=ALL              # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bajoshi@asu.edu  # send-to address

module purge    # Always purge modules to ensure a consistent environment

module load anaconda2/5.2.0

python cluster_full_run_pool.py