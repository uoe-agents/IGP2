(goal_recognition)=
# Goal Recognition

IGP2 can be used solely for goal recognition as well, without needing to simulate for motion planning. 

This page describes how IGP2 can be used for goal recognition on two real-world datasets: [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/).

Note, for reproducing baselines results in the [GRIT repository](https://github.com/uoe-agents/GRIT) (an affiliated project with IGP2) goal recognition should best be run with the 0.2.0 version of IGP2.

## Data
The goal recognition module of IGP2 can be run on existing data sets without the need for CARLA.
Currently, we support [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) to be used to tune and evaluate the goal recognition algorithm of IGP2.
The contents of the data subdirectories in each of these datasets should be moved into `scenarios/data/ind` and `scenarios/data/round`, respectively.

## Running Goal Recognition Experiments

The experiment_multi_process.py scripts allows to run the IGP2 goal recognition in a highly parallelised way.

To run, the script has the following requirements:
- scripts/experiments/data/logs folder exists to store log data
- scripts/experiments/data/results folder exists to store results binary
- .csv files to select with vehicle id and frames to perform goal recognitions at, for each recording, located in the scripts/experiments/data/evaluation_set folder. These csv files can be obtained by run the script `core/data_processing.py` available in the [GRIT repository](https://github.com/uoe-agents/GRIT).

The scripts have the following command line arguments:
- num_workers: number of cpus to use
- output: output filename for the result binary (.pkl extension automatically added)
- tuning: to decide between using the default cost factors or the ones specified in the script
- reward_scheme: which reward scheme to use for likelihood generation
- dataset: run on the validation or test dataset
- h: get description of all command line arguments. Please use this options for more details.

## Running an Experiment on SLURM
To run an experiment on a SLURM enabled server, first add the SBATCH_NUM_PROC variable to your .bashrc. It can be changed depending on how many processors you want to use on the server.

`export SBATCH_NUM_PROC=128`

Navigate to the igp2-dev folder and start an experiment by running:

`nohup sbatch --cpus-per-task=$SBATCH_NUM_PROC scripts/experiments SLURM_experiment.sh &`

Once the experiment is completed, the result binary can be accessed in the scripts/experiments/data/results folder

## Visualisation
The visualisation gui located in ```gui/run_result_track_visualization.py``` is a modified version of the code in https://github.com/ika-rwth-aachen/drone-dataset-tools. It has the following additional features

- It can load a result binary file to display the goal probabilities for each vehicles. It will display the data from the closest frame to the current frame for which results were computed.
- When clicking on a vehicle, it will display the planned trajectory from initial position in cyan and the planned trajectory from current position in green, for all goals.
- When clicking on the vehicle, a new window will appear, plotting the different quantities used for likelihood calculation of the true goal of the agent. In cyan is the planned trajectory from initial position. In green, up to the red line, is the current real trajectory data and, after the red line, the planned trajectory from current position. The quantities are plotted against the pathlength of the trajectory and the red line indicates the current position of the vehicle on the pathlength.

You can find a description of the different command line arguments by running `python run_result_track_visualization.py -h`
