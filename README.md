# Interpretable Goal-based Prediction and Planning (IGP2)

A motion planning and prediction system for autonomous driving.

Latest stable version: 0.3.1; Read the [documentation](https://uoe-agents.github.io/IGP2/).

## Project Description

This code-repository contains the open-source implementation of Interpretable Goal-based Prediction and Planning (IGP2) for autonomous driving, based on [Albrecht et al. (ICRA'21)](https://arxiv.org/abs/2002.02277).
If you would like to get a more detailed understanding of IGP2, please see this [blog post](https://agents.inf.ed.ac.uk/blog/interpretable-prediction-planning-autonomous-driving/index.php) for an introduction.

This implementation of IGP2 is powered by the open-source [ASAM OpenDrive 1.6](https://www.asam.net/standards/detail/opendrive/) standard for road layout definition and the similarly open-source simulated driving environment [CARLA 0.9.13](https://carla.org/).
IGP2 also runs without CARLA using a simple 2D simulator, which obfuscates realistic physical simulation in favour of speed and reproducibility.

The goal recognition module of this implementation also supports both the [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) datasets.
This module was also used as a baseline for comparison to the [GRIT](https://arxiv.org/abs/2103.06113) goal recognition method.

## Please cite

If you use this code, please cite
*"Interpretable Goal-based Prediction and Planning for Autonomous Driving"
by Albrecht et al.* [1] published at ICRA 2021:

```text
@inproceedings{albrecht_interpretable_2021,
  title = "Interpretable Goal-based Prediction and Planning for Autonomous Driving",
  author = "Stefano V. Albrecht and Cillian Brewitt and John Wilhelm and Balint Gyevnar and Francisco Eiras and Mihai Dobre and Subramanian Ramamoorthy",
  booktitle = "IEEE International Conference on Robotics and Automation (ICRA)",
  year = "2021"
}
```

### Acknowledgements

1. The igp2.opendrive module is based on the opendriveparser module of Althoff et al. [2]. Their original code is available [here](https://gitlab.lrz.de/tum-cps/opendrive2lanelet).
2. The gui module is based on the inD Dataset Python Tools available on [GitHub](https://github.com/ika-rwth-aachen/drone-dataset-tools).
3. The CARLA visualiser is based on example code provided as part of CARLA [3].

### Remarks

This project contains an implementation of a queryable road-layout Map based on OpenDrive with partial support of the whole standard.
Notably, signals and controllers are currently not supported, and there is no immediate plan to implement them either, as IGP2 does not rely on these features.

New maps can be created for IGP2 using various tools, e.g. [RoadRunner](https://uk.mathworks.com/products/roadrunner.html) from MathWorks.

A useful GUI to visualise the outputs of the goal recognition method is included in the project.

## Documentation

Guidance on how to get started and full documentation of IGP2 is available on a separate [website](https://uoe-agents.github.io/IGP2/) of this repository, as well as under the `docs` folder.

## References

[1] S. V. Albrecht, C. Brewitt, J. Wilhelm, B. Gyevnar, F. Eiras, M. Dobre, S. Ramamoorthy, "Interpretable Goal-based Prediction and Planning for Autonomous Driving", in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2021

[2] M. Althoff, S. Urban, and M. Koschi, "Automatic Conversion of Road Networks from OpenDRIVE to Lanelets," in Proc. of the IEEE International Conference on Service Operations and Logistics, and Informatics, 2018

[3] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, V. Koltun, "CARLA: An Open Urban Driving Simulator" in Proc. of the 1st Annual Conference on Robot Learning, 2017
