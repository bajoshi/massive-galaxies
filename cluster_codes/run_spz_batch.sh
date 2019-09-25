#!/bin/bash

#SBATCH --partition=fn1              # because I can only use the fat node
#SBATCH -n 5                         # number of cores
#SBATCH --time=00-03:00:00           # Max time for task. Format is DD-HH:MM:SS
#SBATCH -o slurm.spz.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.spz.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=ALL              # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bajoshi@asu.edu  # send-to address

module purge    # Always purge modules to ensure a consistent environment

module load anaconda/py2
source activate galaxy2

export MKL_NUM_THREADS=2

parallel -vkj5 'python cluster_full_run_gnuparallel.py {} > par_out{}.txt' ::: $(seq 0 4)