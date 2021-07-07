#!/bin/bash
#SBATCH --job-name=IGP2_GRIT_cost_tuning_angular_velocity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate IGP2
export PYTHONPATH=$HOME/igp2-dev
python ~/igp2-dev/scripts/experiments/experiment_multi_process.py --num_workers 128 --output validset_heading
