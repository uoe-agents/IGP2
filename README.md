# Interpretable Goal-based Prediction and Planning (IGP2)

A planning and prediction system for autonomous driving.

<h1>Table of Contents</h1>

- [Project Description](#project-description)
- [Please Cite](#please-cite)
- [Acknowledgements](#acknowledgements)
- [Documentation](#documentation)
  - [Requirements](#1-requirements)
  - [Installation](#2-installation)
  - [Data](#3-data)
  - [Running Goal Recognition Experiments](#4-running-goal-recognition-experiments)
  - [Running Simulations in CARLA](#5-running-simulations-in-carla)
- [References](#references)

## Project Description

This code repository contains the open-source implementation of Interpretable Goal-based Prediction and Planning (IGP2) for autonomous driving, based on [Albrecht et al. (ICRA'21)](https://arxiv.org/abs/2002.02277). See this [blog post](https://agents.inf.ed.ac.uk/blog/interpretable-prediction-planning-autonomous-driving/index.php) for an introduction to the IGP2 system.

This implementation of IGP2 is powered by the [OpenDrive](https://www.asam.net/standards/detail/opendrive/) standard for road layout definition and the open-source simulated driving environment [CARLA](https://carla.org/). 

The goal recognition module of this implementation supports both the [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) datasets. This module was also used as a baseline for comparasion to the [GRIT](https://arxiv.org/abs/2103.06113) goal recognition method.

## Please cite:
If you use this code, please cite
**"Interpretable Goal-based Prediction and Planning for Autonomous Driving"
by Albrecht et al. [1] published at ICRA 2021:**

```
@inproceedings{albrecht_interpretable_2021,
  title = "Interpretable Goal-based Prediction and Planning for Autonomous Driving",
  author = "Stefano V. Albrecht and Cillian Brewitt and John Wilhelm and Balint Gyevnar and Francisco Eiras and Mihai Dobre and Subramanian Ramamoorthy",
  booktitle = "IEEE International Conference on Robotics and Automation (ICRA)",
  year = "2021"
}
```

## Acknowledgements
1. The igp2.opendrive module is based on the opendriveparser module of Althoff et al. [2]. Their original code is available [here](https://gitlab.lrz.de/tum-cps/opendrive2lanelet).
2. The gui module is based on the inD Dataset Python Tools available on [GitHub](https://github.com/ika-rwth-aachen/drone-dataset-tools).
3. The CARLA visualiser is based on example code provided as part of CARLA [3].

<hr />

This project contains an implementation of a queryable road-layout Map based on ASAM OpenDrive with partial support of the whole standard.
Notably, our implementation of the Map does not support signals at the moment.
New maps can be created for IGP2 using various tools, e.g. [RoadRunner](https://uk.mathworks.com/products/roadrunner.html) from MathWorks. 

A useful GUI to visualise the outputs of the goal recognition method is included in the project.

## Documentation

### 1. Requirements
Python 3.8 is required.

For a detailed list of package requirements please see the file [requirements.txt](https://github.com/uoe-agents/IGP2/blob/main/requirements.txt).

### 2. Installation
To install the most recent stable release first clone the main branch of the repository. Then, install the python package and the necessary requirements with pip.

```
git clone https://github.com/uoe-agents/IGP2.git
cd IGP2
pip install -e .
```

If you want to, you can checkout the active development branch using `git checkout dev` to access the latest features.

You should now be able to import IGP2 into your existing code base by typing `import igp2 as ip` at the top of your files.

#### Possible Issues:
1. ```FileNotFoundError: .../geos_c.dll (or one of its dependencies)``` - If using conda to manage your environment then try running the following command with your environment activated: ```conda install geos```
2. Running on Mac: CARLA is currently not supported on MacOS, so you won't be able to use IGP2 with CARLA on a Mac.
### 3. Data
The goal recognition module of IGP2 can be run on existing data sets without the need for CARLA.
Currently, we support [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) to be used to tune and evaluate the goal recognition algorithm of IGP2.
The contents of the data subdirectories in each of these datasets should be moved into `scenarios/data/ind` and `scenarios/data/round`, respectively.

### 4. Running Goal Recognition Experiments

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

#### Running an Experiment on SLURM
To run an experiment on a SLURM enabled server, first add the SBATCH_NUM_PROC variable to your .bashrc. It can be changed depending on how many processors you want to use on the server.

`export SBATCH_NUM_PROC=128`

Navigate to the igp2-dev folder and start an experiment by running:

`nohup sbatch --cpus-per-task=$SBATCH_NUM_PROC scripts/experiments SLURM_experiment.sh &`

Once the experiment is completed, the result binary can be accessed in the scripts/experiments/data/results folder

#### Visualisation
The visualisation gui located in ```gui/run_result_track_visualization.py``` is a modified version of the code in https://github.com/ika-rwth-aachen/drone-dataset-tools. It has the following additional features

- It can load a result binary file to display the goal probabilities for each vehicles. It will display the data from the closest frame to the current frame for which results were computed.
- When clicking on a vehicle, it will display the planned trajectory from initial position in cyan and the planned trajectory from current position in green, for all goals.
- When clicking on the vehicle, a new window will appear, plotting the different quantities used for likelihood calculation of the true goal of the agent. In cyan is the planned trajectory from initial position. In green, up to the red line, is the current real trajectory data and, after the red line, the planned trajectory from current position. The quantities are plotted against the pathlength of the trajectory and the red line indicates the current position of the vehicle on the pathlength.

You can find a description of the different command line arguments by running `python run_result_track_visualization.py -h`

### 5. Running Simulations in CARLA 

The `scripts/experiments/carla_traffic_manager.py` script allows the full IGP2 method to be run in the [CARLA simulator](https://carla.org/). At present this is configured to run on the "Town01" map provided with CARLA.
Since IGP2 does not rely on external signals from OpenDrive, the map has to be modified to include junction priorities. 
The version of "Town01" that comes in this repository already contains junction priorities.

This script requires [CARLA](https://carla.org/) 0.9.12 or later to be installed, along with the CARLA python API.

The CARLA server should either already be running in the background when running the above command, or you can pass the ```--launch_process``` command line argument to spawn a new CARLA process. 
If the location of CARLA is not found on the default paths (C:\\Carla on Windows; /opt/carla-simulator on Linux) then the `--carla_path` command line argument can be used to specify the installation location of CARLA.

A description of all command-line options can be found by running ```python carla_traffic_manager.py -h```

## References
[1] S. V. Albrecht, C. Brewitt, J. Wilhelm, B. Gyevnar, F. Eiras, M. Dobre, S. Ramamoorthy, "Interpretable Goal-based Prediction and Planning for Autonomous Driving", in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2021

[2] M. Althoff, S. Urban, and M. Koschi, "Automatic Conversion of Road Networks from OpenDRIVE to Lanelets," in Proc. of the IEEE International Conference on Service Operations and Logistics, and Informatics, 2018

[3] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, V. Koltun, "CARLA: An Open Urban Driving Simulator" in Proc. of the 1st Annual Conference on Robot Learning, 2017
