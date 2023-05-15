(first_steps)=
# First steps

TLDR: Run `python scipts/run.py --map Town01 --carla` from the root directory of IGP2 (or remove `--carla` to use IGP2 with a simpler simulator).

This page will walk you through the steps to run a pre-defined scenario using IGP2. 
It first describes how a scenario is defined on a high-level and then gives a step-by-step guide on how to run scenarios.
More information about how to create your own scenarios and how to run them are available in the ["Custom scenarios"](custom_scenarios.md) page.
The end of this document also contains a summary of all the options you can use to customise the running of a scenario.


## How are scenarios defined?

Scenarios are defined by two main components:

**Road layout**: This is the underlying map of the scenario which contains important semantic annotations used by IGP2. The road layout is defined in the [ASAM OpenDrive 1.6](https://www.asam.net/standards/detail/opendrive/) format. Road layout should be put in the ```scenarios/maps``` folder by default, though there are options to specify another location.

When making your own maps, you should make sure that it contains all information required by IGP2. These are detailed in the ["Road layouts"](road_layout.md) page. 

**Scenario configuration file**: The scenario config file in JSON Schema Version 7. This file contains all the information about how to run the scenario and with what kind of agents. The scenario config file should have the same name as the corresponding road layout, and it should be placed in the ```scenarios/configs``` folder.

Detailed documentation of how to write your own configuration file is given in the ["Configuration files"](configuration_file.md) page.

After both the road layout and the configuration file has been defined, you can follow the next steps to actually run the scenario.

## Running a scenario
To run a scenario using IGP2 you can use the built-in ```scripts/run.py``` script, which provides an extensive interface to the current capabilities supported by IGP2. 

**Important: Make sure you run all scripts from the root directory of IGP2, otherwise relative paths will not work properly.**

To run the Town01 scenario in CARLA you should take the following steps:
1. Start the CARLA server. Either
   1. Start your own server before continuing to step 2; OR
   2. Append the ```--launch_process``` command line option in the next step. This may also require you to specify your CARLA installation location using the ```--carla_path``` command line option.
2. Run the following command: ```python scripts/run.py --map Town01 --carla```.

This will set your CARLA server to synchronous mode (this might make it look like the simulation is frozen while the world is updated), spawn agents into the CARLA world, and run until the ego vehicle has reached its goal, or we have reached the iteration time limit.

It will also open up a separate pygame window, which will display the CARLA simulation from behind the ego vehicle, with additional HUD information relating to telemetries and goal recognition.

<hr />

If you wish to run the same scenario but without CARLA, using our simple simulator, then you can just remove the ```--carla``` command line option and run the same command above. However, you may want to add the ```--plot 20``` option to plot the state of the world every 20 timesteps, as otherwise the simple simulator will not display any visual feedback. 

In summary, the command to run Town01 without CARLA using the simple simulator and plotting is:
```python scenarios/run.py --map Town01 --plot 20```

Full documentation of all available command line options and their effects are given in the page ["Commandline options"](commandline_options.md) or using the command ```python scripts/run.py -h```.

## What scenarios are available by default?

By default, IGP2 comes with multiple scenarios:
1. Heckstrasse: Part of the [inD dataset](https://www.ind-dataset.com/). It is a simple angled T-junction with an unprotected left turn and a separate turning lane.
2. Frankenberg: Part of the inD dataset. An urban four-way crossing with no priorities on any roads and one pedestrian zebra crossing.
3. Bendplatz: Part of the inD dataset. An urban four-way crossing with a main road with priority and unprotected left turns.
4. Neuweiler: Part of the [rounD dataset](https://www.round-dataset.com/). A four-way roundabout with two lanes, by-pass roads and one and two-laned incoming roads.
5. Scenario[1-4]: Part of IGP2. These are the scenarios that IGP2 was tested on.
   1. Scenario 1: T-junction with two lanes.
   2. Scenario 2: Four-way crossing with a priority road.
   3. Scenario 3. Three-way roundabout with two lanes.
   4. Scenario 4. A four-way crossing and a T-junction with a jam of vehicles.
6. Town01: Part of [CARLA](https://carla.readthedocs.io/en/latest/tuto_first_steps/#choose-your-map). A small, simple town with only T-junctions.

[Next: Custom scenarios](custom_scenarios.md)