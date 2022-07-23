#!/bin/bash
#SBATCH --account=rrg-mtaylor3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000M
#SBATCH --time=36:10:00

#SBATCH --array=0-30



cd /home/cmuslima/projects/def-mtaylor3/cmuslima/RT



module load StdEnv/2018 python/3
source RT_env/bin/activate
module load mpi4py/3.0.3
module load scipy-stack
start_time=`date +%s`
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"
python3 config.py --num_runs_start $SLURM_ARRAY_TASK_ID
end_time=`date +%s`
runtime=$((end_time-start_time))

echo "run time"
echo $runtime