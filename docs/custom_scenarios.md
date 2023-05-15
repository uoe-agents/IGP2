(custom_scenarios)=
# Custom scenarios

While IGP2 comes with a number of pre-defined scenarios, creating your own scenarios is where the true power of IGP2 can be put to the test. 
Creating and running a scenario involves three main steps, though both of these steps can take some time to complete:
1. Create a road layout
2. Create a configuration file
3. Run you scenario

## Creating a road layout

Road layouts are at the core of IGP2 providing important semantic information about the roads.

You can follow our guidance in the ["Road layouts"](road_layout.md) page on how to get started with creating a road layout suitable for IGP2.

Once you are done creating your road layout and you have added all the necessary information as described in the linked page above, you should copy it to the `scenarios/maps` folder. 

**If you have built CARLA from source**, you can also import your new map into CARLA which makes it available for editing in Unreal Engine 4, such that you can add buildings, props, terrains, and other decorative elements to the scenario. 
Importing your map into CARLA will also make it available to use with IGP2.

**If you have not built CARLA from source**, then IGP2 uses the built-in methods of CARLA to generate a very simple and minimalist map for CARLA to use in simulation.
Note, that this will only include a very limited set of geometries without any of the additional content that CARLA has to offer.

**If you do not use CARLA**, then it is enough to copy the road layout to the `scenarios/maps` folder. 

Once your road layout definition is accessible by IGP2, you can define your configuration file.


## Creating a configuration file

Configuration files describe important metadata about the scenario and what and how agents will behave in it.

Detailed information about what a configuration files is and the various possible fields it can contain are given in the ["Configuration files"](configuration_file.md) page.

There are some steps to defining your own scenario configuration file:
1. Generate a template
2. Specify scenario metadata
3. Describe agents

### Generate a template

You do not have to create your own scenario file from scratch.

Run the following command to have one created for you. You can also use this command to specify the spawn and goal positions of agents by simply clicking on the map of the road layout:
```bash
python scripts/genconfig.py
```

This command on its own, without any command line options, will create a scenario configuration named `scenarios/configs/new_scenario.json` with one ego agent and one traffic agent for a map called `new_scenario.xodr`.
However, you can specify your own options via command line arguments as follows:
- `-n, --name NAME`: The name of your scenario. NAME should be a string.
- `-nm, --n_mcts N_MCTS`: The number of MCTSAgents to add to the configuration file. N_MCTS should be a natural number.
- `-nt, --n_traffic N_TRAFFIC`: The number of TrafficAgents to add to the configuration file. N_TRAFFIC should be a natural number.
- `-o, --output_path PATH`: Directory to save the generated configuration file to. PATH should be a string that points to a valid directory.
- `--set_locations`: If given, then display a map of the road layout to select the positions of spawns and goals more easily. 

You can also access these options and their descriptions via `python scripts/genconfig.py -h`.

## Specify scenario metadata

Once your template has been generated you can specify the metadata used for your simulation.
The two most important to specify are `map_path` and `max_steps`, however you can alter almost all global parameters of IGP2. Please consult the properties of the Configuration class in the file `igp2/config.py` to see which parameters can be modified.

When using CARLA, you can also specify the `--n_traffic N_TRAFFIC` keyword as an integer.
If this option is given then N_TRAFFIC number of agents will be randomly spawned dynamically into the environment.
Note, N_TRAFFIC will include the number of agents hand-defined in the configuration file as well.
For example, if N_TRAFFIC is 10 and there are 3 non-ego agents in the world defined by you, then the TrafficManager will spawn 7 more agents.

## Describe agents

The most important part of the configuration file is to describe the agent that are to appear in the scenario.

Currently, two types of agents are supported by IGP2:
1. MCTSAgent: The MCTS Agent use the full IGP2 system for goal recognition and planning to drive around the road. These are generally used as the ego agent of the scenario.
2. TrafficAgent: The Traffic Agent is a simpler, path-following agent that drives from a given spawn location to its goal.

**Both types of agents** require you to define their ID, spawn box, starting velocity range, and goal box.
IDs can be any unique integer from the other agents.
Boxes are two dimensional (i.e., rectangles) defined via a center point, length, width, and heading (i.e., orientation).
Heading is defined using radians following the orientation of the standard [unit circle](https://en.wikipedia.org/wiki/Unit_circle#/media/File:Unit_circle_angles_color.svg).

To find a suitable position for these parameters, you can quickly plot the road layout using matplotlib via our function `plot_map` located in `igp2/opendrive/plot_map.py` or you can pass the `--map MAP_NAME_TO_PLOT --plot_map_only` command line arguments to `scripts/run.py` to plot the map.

**Traffic agents** can optionally take a list of `macro_actions` that describe the sequence of macro actions to be executed rather than following a generated path from the spawn to the goal.
Note, if you use this option, then you should make sure that the sequence of given macro actions do actually take the agent to the specified goal.

**MCTS agents** take optional parameters for changing how trajectory costs and rewards are calculated when using the IGP2 system.
They can also take a `view_radius` option, which limits the view distance of the agent to a circle centered on the current position of the agent with the specified radius.
IGP2 also works with dynamically generated stopping goals. If you wish to disable this, you can alter the `stop_goals` parameter.
