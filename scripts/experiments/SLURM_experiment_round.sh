#!/bin/bash
#SBATCH --job-name=IGP2
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate IGP2
export PYTHONPATH=$HOME/igp2-dev
python ~/igp2-dev/scripts/experiments/experiment_multi_process_round.py --num_workers $SBATCH_NUM_PROC --output validset_round --dataset valid --tuning 1 --reward_scheme 1
