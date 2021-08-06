# IGP2

TODO

The igp2.opendrive module is based on the opendriveparser module  of Althoff, et al. [1]. Their original code is available here: https://gitlab.lrz.de/tum-cps/opendrive2lanelet

<hr />

## Documentation

### Running an experiment

The experiment_multi_process.py scripts allows to run the IGP2 goal recognition in a highly parallelised way.

To run, the script has the following requirements:
- scripts/experiments/data/logs folder exists to store log data
- scripts/experiments/data/results folder exists to store results binary
- .csv files to select with vehicle id and frames to perform goal recognitions at, for each recording, located in the scripts/experiments/data/evaluation_set folder.

The scripts has the following command line arguments:
- num_workers: number of cpus to use
- output: output filename for the result binary (.pkl extension automatically added)
- tuning: to decide between using the default cost factors or the ones specified in the script
- reward_scheme: which reward scheme to use for likelihood generation
- dataset: run on the validation or test dataset
- h: get description of all command line arguments. Please use this options for more details.

### Running an experiment on the server
To run an experiment on a SLURM enabled server, first add the SBATCH_NUM_PROC variable to your .bashrc. It can be changed depending on how many processors you want to use on the server.

`export SBATCH_NUM_PROC=128`

Navigate to the igp2-dev folder and start an experiment by running:

`nohup sbatch --cpus-per-task=$SBATCH_NUM_PROC scripts/experiments SLURM_experiment.sh &`

Once the experiment is completed, the result binary can be accessed in the scripts/experiments/data/results folder

### Visualisation
The visualisation gui located in gui/run_result_track_visualization.py is a modified version of the code in https://github.com/ika-rwth-aachen/drone-dataset-tools. It has the following additional features

- It can load a result binary file to display the goal probabilities for each vehicles. It will display the data from the closest frame to the current frame for which results were computed.
- When clicking on a vehicle, it will display the planned trajectory from initial position in cyan and the planned trajectory from current position in green, for all goals.
- When clicking on the vehicle, a new window will appear, plotting the different quantities used for likelihood calculation of the true goal of the agent. In cyan is the planned trajectory from initial position. In green, up to the red line, is the current real trajectory data and, after the red line, the planned trajectory from current position. The quantities are plotted against the pathlength of the trajectory and the red line indicates the current position of the vehicle on the pathlength.

You can find a description of the different command line arguments by running `python run_result_track_visualization.py -h`

### Analysis

Note: this script will not be part of the official release, or should be reworked. The documentation below is for internal use only.

A rough script for analysis is provided to perform checks on the results over the whole datasets in scripts/experiments/cost_tuning_analysis.py

Roughly, the script performs the following actions

- Loads a result binary, decides on which results to perform analysis on.
- if REMOVE_UNCOMPLETED_PATH is set to True, remove any agent_result that is incomplete (less than 11 points).
- if REMOVE_UNFEASIBLE_PATHS is set to True, remove any agent_result that does not compute a feasible path to its final goal.
- experiment 4: prints agents who have a planned trajectory duration from their current point of over 30 s.
- experiment 0: print which agents have unfeasible paths to their true goals, for inputted "spikes", which represents indices of the agent_result class for each scenarios.
- experiment 1: outputs the percentage of unfeasible true goals in the results, and splits them according to each goal, for debugging purposes.
- experiment 2: calculates and plots the average goal probability associated to the true goal for each scenario.
- experiment 3: calculates and plots the average goal accuracy (% chance of the true goal being the most likely predicted goal) for each scenario.

## References
[1] M. Althoff, S. Urban, and M. Koschi, "Automatic Conversion of Road Networks from OpenDRIVE to Lanelets," in Proc. of the IEEE International Conference on Service Operations and Logistics, and Informatics, 2018

