#!/bin/bash
#
#SBATCH --job-name=IGP2_GRIT_cost_tuning_acceleration
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=124
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3.5G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate IGP2
python ~/igp2-dev/scripts/experiments/experiment_multi_process.py --num_workers 12 --output angular_velocity_tuning