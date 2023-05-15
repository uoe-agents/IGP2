(configuration_file)=
# Configuration files

Configuration files are essential to running scenarios with IGP2.
They describe, among others, the location of the road layout to use, the length of the simulation, global parameters (e.g., speed limits, sampling distances, etc.).
Most importantly, they also contain a parametrisation of the various agents that should be spawned into the simulation, their goals, and their behaviour.

Configuration files should have JSON Schema Version 7 and they should end in the `.json` file extension.
By default, they should be placed into the `scenarios/configs` folder to make them accessible via the `--map` command line option.

This page describes in detail the structure of configuration files, including all available fields.

## Configuration file structure
You can generate a template configuration file with the following command: `python scripts/genconfig.py NAME`, where is the name of the configuration file, as in `NAME.json`. Further command line options are available through the `-h` option.

Some additional remarks:
1. Fields with an exclamation mark (!) must always be included.
2. Angles should be in radians and ideally in the range [-pi,pi].
3. Setting many of these parameters to an appropriate value can often be non-trivial (e.g., cost weight and MCTS reward factors). You can consult the :ref:`IGP2 API <api>` to get a better understand of what each value represents.

The following gives all possible field names, types, and descriptions, as well as their intended hierarchical relationship.

```text
scenario: dict      Dictionary of global options and parameters
|    |____!map_path: str    The path to the OpenDrive *.xodr file for the road layout
|    |____fps: int          Execution frequency of the simulation
|    |____seed: int         Random seed
|    |____max_steps: int    If given, stop executing simulation after this many steps
|    |____n_traffic: int    The number of vehicles to be managed by the TrafficManager when using CARLA
|    |____**Any global parameters supported by the Configuration class in the igp2/config.py.
```
```text
--------------The following is supported by all Agents-----------------
|____agents: list       List of agents to spawn
|    |____agent: dict       Dictionary of agent to spawn
|    |    |____!id: int        The numerical identifier of the agent
|    |    |____!type: str      The type of the agent. Currently, only MCTSAgent and TrafficAgent is supported
|    |    |____!spawn: dict             The spawn area of the vehicle
|    |    |   |____!box: dict                The box to spawn the vehicle in
|    |    |   |   |____!center: list[float]      The center of the spawn area
|    |    |   |   |____!length: float            The length of the spawn area
|    |    |   |   |____!width: float             The width of the spawn area
|    |    |   |   |____!heading: float           The heading of the spawn area
|    |    |   |____velocity: list[float]     The spawn velocity range for random sampling
|    |    |____!goal: dict               The goal area of the vehicle
|    |    |   |____!box: dict                The box for the vehicle to reach
|    |    |   |   |____!center: list[float]      The center of the goal area
|    |    |   |   |____!length: float            The length of the goal area
|    |    |   |   |____!width: float             The width of the goal area
|    |    |   |   |____!heading: float           The heading of the goal area
|    |    |____goal_recognition: dict     Any keyword arguments supported by the GoalRecognition class in igp2/recognition/goalrecognition.py
|    |    |____velocity_smoother: dict    Any keyword arguments supported by the VelocitySmoother class in igp2/velocitysmoother.py
```
```text
-----------The following is supported by TrafficAgents only---------------
|    |    |____macro_actions: list             An ordered list of macro actions to be executed by the agent
|    |    |    |____macro_action: dict             Configuration dict of a macro action
|    |    |    |    |____type: str                     Type of the macro action
|    |    |    |    |____**Any other keyword arguments supported by the given macro action type above.
```
```text
-----------The following is for MCTSAgents (and subclasses) only-----------
|    |    |____cost_factors: dict           The cost weighing factors for IGP2 goal recognition
|    |    |    |____time: float                  Time to goal
|    |    |    |____velocity: float              Average velocity
|    |    |    |____acceleration: float          Average acceleration
|    |    |    |____jerk: float                  Average jerk
|    |    |    |____heading: float               Average heading
|    |    |    |____angular_velocity: float      Average angular velocity
|    |    |    |____angular_acceleration: float  Average angular acceleration
|    |    |    |____curvature: float             Average curvature
|    |    |    |____safety: float                Safety of trajectory
|    |    |____mcts: dict                  MCTS parameters
|    |    |    |____t_update: float                Runtime period of MCTS in seconds
|    |    |    |____n_simulations: int             Number of rollouts in MCTS
|    |    |    |____store_results: str             Either 'all' or 'final'. If absent, results are not stored
|    |    |    |____trajectory_agents: bool        Whether to use trajectory following or macro action executing agents
|    |    |    |____reward_factors: dict  MCTS reward factors
|    |    |    |    |____time: float                    Time to goal
|    |    |    |    |____jerk: float                    Average jerk
|    |    |    |    |____angular_velocity: float        Average angular velocity
|    |    |    |    |____curvature: float               Average curvature
|    |    |    |    |____safety: float                  Safety term. Currently, not used.
|    |    |    |____**Any other keyword argument supported by the MCTS class in igp2/planning/mcts.py.
|    |    |____view_radius: float         Radius of circle centered on the vehicle, in which it can observe the environment
|    |    |____stop_goals: bool           Whether the agent should include stopping goals for goal recognition.                           
```