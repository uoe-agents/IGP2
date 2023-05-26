(commandline_options)=
# Commandline options

This page documents all command line options available to use with scripts.
The below information can be obtained by adding the `-h` flag after each script call.

## run.py
- `--map, -m MAP`: The name of the scenario to run. Must be specified. MAP should be a string of an existing scenario in the ```scenarios``` folder.
- `--config_path PATH`: Alternatively to ```--map``` you can specify the direct file of the configuration file using this argument. PATH should be a string that points to a valid configuration file.
- `--carla`: If present, then run the scenario simulation using CARLA, otherwise rely on IGP2's built-in simple simulator.
- `--seed SEED`: The random seed of the simulation. SEED should be a whole number.
- `--fps FPS`: The execution framerate of the simulation. FPS is set to 20 by default. FPS should be an integer.
- `--debug`: If present, then display debugging logging statements and plots.
- `--save_log_path PATH`: The directory to where the runtime log should be saved. Disabled by default. PATH should be a string that points to a valid folder.
- `--plot PLOT_T`: If given, the period of plotting the state of the simulation. PLOT_T should be an integer.
- `--server SERVER`: The server IP address where the CARLA server can be reached. Set to "localhost" by default. SERVER should be a string.  
- `--port PORT`: The port through which the CARLA server is accessible under the given server IP address. PORT is 2000 by default. PORT should be an integer.
- `--carla_path, -p PATH`: The installation folder of CARLA. Set to ```/opt/carla-simulator``` by default. PATH should be a string that points to a valid folder.
- `--launch_process`: If present, then a new CARLA process will be launched using the CARLA executable pointed to by ```--carla_path, -p```.
- `--no_rendering`: If present, then CARLA rendering will be disabled.
- `--no_visualiser`: If present, then the pygame visualiser will not be displayed.
- `--record, -r`: If present, then the CARLA simulation will be recorded for playback.
- `--plot_map_only`: If present, then only plot the given map and then exit the program.
- `--record_visualiser`: If present then the PyGame visualiser window will be saved frame-by-frame to the folder `scripts/experiments/data/pygame_recordings`. Any other previous recording in this folder will be overriden.

## genconfig.py
- `-n, --name NAME`: The name of your scenario. NAME should be a string.
- `-nm, --n_mcts N_MCTS`: The number of MCTSAgents to add to the configuration file. N_MCTS should be a natural number.
- `-nt, --n_traffic N_TRAFFIC`: The number of TrafficAgents to add to the configuration file. N_TRAFFIC should be a natural number.
- `-o, --output_path PATH`: Directory to save the generated configuration file to. PATH should be a string that points to a valid directory.
- `--set_locations`: If given, then display a map of the road layout to select the positions of spawns and goals more easily. 
