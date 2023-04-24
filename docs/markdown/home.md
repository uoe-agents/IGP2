# Documentation of Interpretable Goal-based Prediction and Planning (IGP2)

Welcome to the Interpretable Goal-based Prediction and Planning (IGP2) documentation.

This home page contains an index of contents of the full documentation of IGP2.
If you are new to using IGP2, we recommend to start at with the below three steps, however, feel free to read the documentation in any order that suits you.

1. Install IGP2: You should check out the "[Installation](installation.md)" page.
2. Start using IGP2: You can refer to the "[First steps](first_steps.md)" page to run IGP2.
3. Create your own scenario: You can create your own scenario by following the "[Custom scenarios](custom_scenarios.md)" page.
4. Refer to the API: If you would like to develop your own code using IGP2 then you can consult the "[Python API](api.md)".

## Support

If you have any issues with installing or running the code, then feel free to use the [GitHub Issues](https://github.com/uoe-agents/IGP2/issues) page to ask for help. We will endeavour to answer questions as soon as possible.

### Bug reports
If you encounter a bug with IGP2 please put in a ticket to the GitHub issues page with your systems setup information,  steps about how to reproduce your bug, and the debug log from your running of IGP2.

## Getting started

**[Installation](installation.md)** - This page guides you through all the necessary steps to get up and running with IGP2.

**[First steps](first_steps.md)** - This page guides you through how to run your first driving scenario with IGP2.


## Components

**[Goal recognition](goal_recognition.md)**: The goal recognition module of IGP2 can be run in standalone mode without simulating for motion planning. This page tells you how to do just that.

**[Creating your own scenarios](custom_scenarios.md)**: IGP2 comes with several pre-defined scenarios, but this page will guide you through how you can create your own scenarios.

**[Road layouts](road_layout.md)**: IGP2 relies on the OpenDrive 1.6 standard for road layout definition, however there are some important additional steps you should take when creating your own maps for IGP2. This page will walk you through those steps.

**[Configuration files](configuration_file.md)**: This page gives full documentation of all fields that can appear in configuration files of IGP2.

## Extending IGP2

**[Add your own macro actions and maneuvers]()**: You can add new agent behaviour to IGP2 by defining new macro actions and maneuvers.

**[Overwriting default MCTS behaviours]()**: You can also overwrite how MCTS is run to add custom planning behaviour.

## Resources

**[Commandline options](commandline_options.md)**: A list of commandline options to be used with `scripts/run.py`.

**[Python API](api.md)**: Directory of the API of IGP2 organised according to its package hierarchy.

**[Useful links](useful_links.md)**: A collection of useful links relevant to working with IGP2.
