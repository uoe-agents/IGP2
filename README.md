# IGP2

This repo contains the open-source implementation of the method described 
in the paper:

"Interpretable Goal-based Prediction and Planning for Autonomous Driving"
by Albrecht, et al [1] published at ICRA 2021: https://arxiv.org/abs/2002.02277

The code in this repository is written and maintained by Cillian Brewitt (@cbrewitt), Balint Gyevnar (@gyevnarb) and Samuel Garcin (@francelico)

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

The igp2.opendrive module is based on the opendriveparser module of Althoff et al. [1]. Their original code is available here: https://gitlab.lrz.de/tum-cps/opendrive2lanelet
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
git clone https://github.com/uoe-agents/IGP2
cd IGP2
pip install -e .
```

#### Possible Issues:
1. ```FileNotFoundError: .../geos_c.dll (or one of its dependencies)``` - If using conda to manage your environment then try running the following command with your environment activated: ```conda install geos```
### 3. Data

The [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) datasets can be used to train and evaluate IGP2.
The contents of the data subdirectories in each of these datasets should be moved into `scenarios/data/ind` and `scenarios/data/round` respectively.

### 4. Running goal recognition experiments with IGP2

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

### 5. Running Experiments in CARLA 

The `carla_traffic_manager.py` script allows the full IGP2 method to be run in the [CARLA simulator](https://carla.org/). At present this has only configured for the "town01" map.

This script requires [CARLA](https://carla.org/) 0.9.11 or later to be installed, along with the CARLA python API. The install location of CARLA should be passed to the script using the `--carla_path` command line argument.

There are several know existing issues when running experiments in CARLA:
1. Sometimes the ego vehicle and another vehicle both enter a junction at the same time and end in a deadlock.
2. The ego vehicle turns erratically and crashes in the junction.
3. When performing goal recognition for other vehicles, sometimes all possible goals are found to be unreachable during A* search.

## Notes

### Differences in implementation from original IGP2 paper

An alternative method of computing likelihoods is available in this implementation. Instead of computing the likelihood as the difference of rewards associated with two trajectories, the likelihood is computed as 

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=L(s_{1:t} | G^i) = \exp( \beta (\Delta r) )">
</p>

where

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\Delta r = \sum_{k=2}^K w_k \frac{1}{N} \sum_{n=1}^N |\hat{r}_{k_n} - \bar{r}_{k_n}|">
</p>

The trajectories are resampled along their path length to <img src="https://render.githubusercontent.com/render/math?math=N"> points (except for the time to goal, as it is a scalar, and there a simple difference is taken). Each individual reward term <img src="https://render.githubusercontent.com/render/math?math=\Delta r_k = \frac{1}{N} \sum_{n=1}^N |\hat{r}_{k_n} - \bar{r}_{k_n}|"> represents the degree of similarity between the two trajectories of the <img src="https://render.githubusercontent.com/render/math?math=k^{th}"> property along the path length. The individual reward terms <img src="https://render.githubusercontent.com/render/math?math=\hat{r}_{k_n}"> are associated to the optimal trajectory from the vehicle's initial observed state to goal <img src="https://render.githubusercontent.com/render/math?math=G^i"> after velocity smoothing, and the individual reward terms <img src="https://render.githubusercontent.com/render/math?math=\bar{r}_{k_n}"> are associated to the trajectory which follows the observed trajectory until time <img src="https://render.githubusercontent.com/render/math?math=t"> and then continues optimally to goal <img src="https://render.githubusercontent.com/render/math?math=G^i">, with smoothing applied only to the trajectory after <img src="https://render.githubusercontent.com/render/math?math=t">.

Essentially, instead of quantifying the similarity between the two trajectories as the difference of the weighted sums of their averaged trajectory properties as in [1], we measure it as the weighted sum of the differences of their individual properties along the trajectories' path length. This change has two purposes. First, it enables to measure the similarity between two vehicle's trajectories as how the trajectories' different physical properties will differ locally as the vehicle is progressing on the road. Second, it permits to easily implement new trajectories properties as reward terms that were not employed in [1]. The properties that we added in this open-source implementation are the heading and the velocity of the vehicles. Since this likelihood calculation strategy led to higher goal prediction accuracy in our experiments, it is enabled by default.

A few other minor changes have been made in this implementation. The objective function used for velocity smoothing is 

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\min_{x_{2:n}, v_{2:n}} \sum_{t=1}^n (v_t - \kappa(x_t))^2 %2B \lambda\sum_{t=1}^{n-1} (v_{t%2B1} - v_t)^2">
</p>


instead of the one presented in [1]. If the optimiser fails to converge, we progressively relax the optimisation constraints and run the smoothing process again. We start by removing the <img src="https://render.githubusercontent.com/render/math?math=v_t \leq \kappa(x_t)"> constraint, following by removing the <img src="https://render.githubusercontent.com/render/math?math=| v_{t%2B1} - v_t| < a_{\max} \Delta t"> and <img src="https://render.githubusercontent.com/render/math?math=v_t \leq v_{max}"> constraints. We then set <img src="https://render.githubusercontent.com/render/math?math=\lambda = 0"> and finally remove the <img src="https://render.githubusercontent.com/render/math?math=v_1 = \hat{v}_1"> constraint. If none of these steps are successful, we return the original velocity profile. In practice, the optimiser always converge after removing one or several constraints.

The reward terms, with the exception of the time to goal, are normalised between 0 and 1 for stricly positive quantities and -1 and 1 for quantities that can be both positive and negative, according to their distributions across both datasets, with values falling beyond three standard deviations of the distribution being clipped. This was done to make the tuning of the reward weights more straightforward.

Lastly, the ego vehicle path planning, the maneuvers prediction module, the closed-loop maneuvers and the stop maneuver are not yet part of this open-source release.

### Acknowledgements

The visualisation code is based on https://github.com/ika-rwth-aachen/drone-dataset-tools

The igp2.opendrive module is based on the opendriveparser module 
of Althoff, et al. [2]. Their original code is available here: https://gitlab.lrz.de/tum-cps/opendrive2lanelet


## References
[1] S. V. Albrecht, C. Brewitt, J. Wilhelm, B. Gyevnar, F. Eiras, M. Dobre, S. Ramamoorthy, "Interpretable Goal-based Prediction and Planning for Autonomous Driving", in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2021

[2] M. Althoff, S. Urban, and M. Koschi, "Automatic Conversion of Road Networks from OpenDRIVE to Lanelets," in Proc. of the IEEE International Conference on Service Operations and Logistics, and Informatics, 2018

