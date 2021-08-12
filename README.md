# IGP2

This repo contains the open-source partial implementation of the method described 
in the paper:

"Interpretable Goal-based Prediction and Planning for Autonomous Driving"
by Albrecht, et al [1] published at ICRA 2021: https://arxiv.org/abs/2002.02277

# Please cite:
If you use this code, please cite
"Interpretable Goal-based Prediction and Planning for Autonomous Driving"
```
@inproceedings{albrecht_interpretable_2021,
title = "Interpretable Goal-based Prediction and Planning for Autonomous Driving",
author = "Albrecht, {Stefano V} and Cillian Brewitt and John Wilhelm and Balint Gyevnar and Francisco Eiras and Mihai Dobre and Subramanian Ramamoorthy",
booktitle = "IEEE International Conference on Robotics and Automation (ICRA)",
year={2021}
}
```

The igp2.opendrive module is based on the opendriveparser module  of Althoff, et al. [1]. Their original code is available here: https://gitlab.lrz.de/tum-cps/opendrive2lanelet
The gui module is based on the inD Dataset Python Tools available at https://github.com/ika-rwth-aachen/drone-dataset-tools

<hr />

This project contains an implementation of a queryable road-layout 
map based on ASAM OpenDrive with partial support of the whole standard. 
(https://www.asam.net/standards/detail/opendrive/) 

A useful GUI to visualise the outputs of the method is also included in the project.

## Documentation

### 1. Requirements
Python 3.8 or later is required.

### 2. Installation
First, clone the repository. Then, install the python package with pip.

```
git clone https://github.com/uoe-agents/GRIT.git
cd IGP2
pip install -e .
```

### 3. Data

The [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) datasets can be used to train and evaluate IGP2.
The contents of the data subdirectories in each of these datasets should be moved into `scenarios/data/ind` and `scenarios/data/round` respectively.

### 4. Running experiments with IGP2

The experiment_multi_process.py scripts allows to run the IGP2 goal recognition in a highly parallelised way.

To run, the script has the following requirements:
- scripts/experiments/data/logs folder exists to store log data
- scripts/experiments/data/results folder exists to store results binary
- .csv files to select with vehicle id and frames to perform goal recognitions at, for each recording, located in the scripts/experiments/data/evaluation_set folder. These csv files can be obtained by run the script `core/data_processing.py` available in the [GRIT repository](https://github.com/uoe-agents/GRIT).

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

## Notes

The igp2.opendrive module is based on the opendriveparser module 
of Althoff, et al. [2]. Their original code is available here: https://gitlab.lrz.de/tum-cps/opendrive2lanelet


## References
[1] SV Albrecht, C Brewitt, J Wilhelm, F Eiras, M Dobre, S Ramamoorthy, "Integrating planning and interpretable goal recognition for autonomous driving", in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2021

[2] M. Althoff, S. Urban, and M. Koschi, "Automatic Conversion of Road Networks from OpenDRIVE to Lanelets," in Proc. of the IEEE International Conference on Service Operations and Logistics, and Informatics, 2018

