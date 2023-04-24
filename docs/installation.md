# Installation

This page walks you through the necessary to installing all pre-requisites of IGP2, including CARLA.
IGP2 is a python package composed of several sub-packages that implement all the functionality of IGP2.
The installation also comes with a GUI tool to visualise goal recognition probabilities and trajectories.


## Before you begin

- **Python**: The minimum required version is [Python 3.9](https://www.python.org/downloads/release/python-3913/).
- **pip**: Pip is necessary to install IGP2, as it is not currently deployed on online package repositories. Any recent version of pip is acceptable.
- **Python pacakges**: The Python package requirements are detailed in the requirements.txt file in the root directory of the code-repository. The installation script installs these automatically.
- **CARLA** (optional): [CARLA 0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13) can be used together with IGP2 to perform simulations, however it is not essential for the running of IGP2. If you do use CARLA, then make sure your computer also meets its [requirements](https://carla.readthedocs.io/en/latest/start_quickstart/).

We recommend you use some kind of virtual environment to install IGP2 to avoid polluting your global Python package environment. 
We use [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).


## Install IGP2
First, navigate to the folder into which you want to install IGP2.

Then, run the following commands:

```
git clone https://github.com/uoe-agents/IGP2.git
cd IGP2
pip install -e .
```

This installs the required Python packages and the current stable **main** branch of IGP2. 

You can check if your installation has been successful by typing ```import igp2 as ip``` into a Python console.


#### Development and older version
If you want to, you can checkout the active development branch using `git checkout dev` to access the latest features.

If you wish to use IGP2 with a previous version you can checkout the corresponding tag using `git checkout TAG` before installing the package.

#### Possible Issues:
1. `FileNotFoundError: .../geos_c.dll (or one of its dependencies)` - You are missing the binaries for GEOS, which is a geometry library. If using Miniconda3 to manage your environment then try running the following command with your environment activated: ```conda install geos```. 
2. Running on MacOS: CARLA is currently not supported on MacOS, so you won't be able to use IGP2 with CARLA on a Mac.


## [Next: First steps](first_steps.md)
