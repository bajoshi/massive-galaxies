#!/bin/bash
BATCH_PATH="${1:?ERROR -- SUBMIT SCRIPT -- Path to BATCH MANIFEST missing}"

BATCH_NAME=$(basename "$BATCH_PATH")
runlog_dir="logs/run"
joblog_dir="logs/joblog"
SLURMlog_dir="logs/SLURM"
joblog="${joblog_dir}/${BATCH_NAME}.log"

SLURM_ERR="${SLURMlog_dir}/slurm.spz.${BATCH_NAME}.%j.err"
SLURM_OUT="${SLURMlog_dir}/slurm.spz.${BATCH_NAME}.%j.out"

for log_dir in $runlog_dir $joblog_dir $SLURMlog_dir; do
  ! [[ -d "$log_dir" ]] && mkdir -p "$log_dir" || :
done

main() {
  input_galaxy="${1:?ERROR -- MAIN -- input galaxy integer missing}"
  log_dir="${2}"
  record_file="${log_dir}/par_out_${input_galaxy}.txt"
  python cluster_full_run_gnuparallel.py $input_galaxy > "$record_file"
}

export -f main

sbatch --comment="galaxies $BATCH_NAME" << EOF
#!/bin/bash
#SBATCH --job-name="$BATCH_NAME"     # because I can only use the fat node
#SBATCH --partition=serial           # because I can only use the fat node
#SBATCH --qos=normal                 # quality of service line
#SBATCH --exclusive                  # empty node
#SBATCH --time=01-00:00:00           # Max time for task. Format is DD-HH:MM:SS
#SBATCH -o $SLURM_OUT                # STDOUT (%j = JobId)
#SBATCH -e $SLURM_ERR                # STDERR (%j = JobId)
#SBATCH --mail-type=ALL              # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bajoshi@asu.edu  # send-to address

# Always purge modules to ensure a consistent environment
module purge    
module load anaconda/py2
source activate galaxy2
export MKL_NUM_THREADS=2
parallel --joblog $joblog --resume -vkj10 main {} $runlog_dir :::: $BATCH_PATH
EOF
