(new_behaviour)=
# Adding new behaviour

You might find yourself in a position where the built-in macro actions and maneuvers of IGP2 are not sufficient for your experiments.
In this case, you can easily create your own classes to support custom behaviour.

## What are macro actions and maneuvers?
*NB.: More details are available in the IGP2 [paper](https://five.ai/igp2).*

Macro actions and their lower-level relatives, maneuvers, are responsible for defining the driving behaviour of every vehicle in the environment.
There is a hierarchical relationship to these elements (from top to bottom):

```text
Macro action --> maneuver (CL) --> maneuver (OL)
```

Maneuvers come in two flavours: open-loop (OL) and closed-loop (CL).

Open-loop maneuvers are the lowest in this hierarchy of actions. 
They define the trajectories and target velocities for vehicles to follow based on the state of the environment.
Using these trajectories and velocities, closed-loop maneuvers use adaptive cruise control and PID controllers to generate acceleration and steering actions that can be executed realistically by the vehicle.

Macro actions are the highest level actions which chain together and parametrise maneuvers, either all OL or all CL, but never mixing OL and CL maneuvers.

IGP2 comes with the following macro actions and maneuvers:

| Macro Action    | Maneuvers                      |
|-----------------|--------------------------------|
| Continue        | follow-lane                    |
| ChangeLaneLeft  | follow-lane, switch-lane-left  |
| ChangeLaneRight | follow-lane, switch-lane-right |
| Exit            | follow-lane, give-way, turn    |
| Stop            | stop                           |

## How to define your own behaviour?

In order to create new behaviour for vehicles you have to create a new macro action.
For this you must do the following three steps:
1. Define any new OL maneuvers
2. Define and register any new CL maneuvers
3. Define and register a new macro action

Before you get started writing your own code, you can take a look at the current implementation of these macro actions and maneuvers which are found in the package `igp2.planlibrary`.

### Creating a new OL maneuver

To create a new OL maneuver, you should create a new class as a subclass of the `igp2.planlibrary.maneuver.Maneuver` parent class. 
All maneuvers (CL or OL) must inherit this class. If your maneuver only involves lane-following then you may also inherit the  `igp2.planlibrary.maneuver.FollowLane` class instead, which provides built-in lane following capabilities.

After creating your class, you should override any methods where you wish to insert new behaviour.

The following code snippet shows how to create a new subclass of `Maneuver` and which methods should be overriden.

```python
import igp2 as ip

class NewOLManeuver(ip.Maneuver):
    """ My new open-loop maneuver. """

    def get_trajectory(self, frame: Dict[int, ip.AgentState], scenario_map: ip.Map) \ 
            -> ip.VelocityTrajectory:
        """ The most crucial method to override. This method determines
            the entire trajectory and velocity profile of the maneuver. """
    
    def applicable(state: ip.AgentState, scenario_map: ip.Map) -> bool:
        """ Important to override. This method determines whether the maneuver
            can be executed in the given state of the environment at all. """
        
    def _get_lane_sequence(self, state: ip.AgentState, scenario_map: ip.Map) \
        -> List[ip.Lane]:
        """ This method returns the sequence of lanes that the vehicle
            should follow during its execution of this maneuver. """
```

Once you have defined at least the `get_trajectory`, `applicable`, `_get_lane_sequence` functions in your OL maneuver, then it is ready to be used in the next step: defining and registering CL maneuvers.

### Creating a new CL maneuver

Closed-loop maneuvers are not much different from OL maneuvers, except that they include controllers to produce physically executable acceleration and steering actions.

The steps to creating a new CL maneuver is very similar to OL maneuvers.
You should create a new class and inherit two classes: your new OL maneuver created above and `igp2.planlibrary.maneuver_cl.WaypointManeuver` (in this order).

You can then add more custom behaviour which will be specific to the CL maneuver only (so not to the OL maneuver from which the CL maneuver inherits).

To add custom behaviour you should overwrite the following methods.
```python
import igp2 as ip

class NewCLManeuver(NewOLManeuver, ip.WaypointManeuver):
    """ My new closed-loop maneuver. """
    
    def next_action(self, observation: Observation) -> Action:
        """ Next action is called on every iteration step of the simulation. 
            Overwrite this method if you wish to change the control of the vehicle."""

    def done(self, observation: Observation) -> bool:
        """ This method is used to check on every iteration step, whether
            the maneuver has terminated. By default, a maneuver terminates 
            when it has reached the final waypoint of its trajectory. """

    def reset(self):
        """ This method is used to reset the internal state of the maneuver.
            If you maneuver tracks some variables and has persistent internal states,
            then you can use this method to reset those variables. """
```

If you do not wish to add custom behaviour that is just specific to the CL maneuver then you can just create an empty class with the appropriate inheritances.

Once you have created your new CL maneuver, you must register it.
To do this, you can add the following lines to your script.
These lines should be called before IGP2 is run.

```python
import igp2 as ip
type_str = "new_cl_maneuver"  # Enter a descriptive short name for your maneuver here
type_man = NewCLManeuver  # The type of your maneuver. I.e., the newly created class without parentheses.
ip.CLManeuverFactory.register_new_maneuver(type_str, type_man)  # Register your new maneuver with IGP2
```

### Creating a new macro action

Macro actions are the core of behaviour in IGP2. 
Both prediction and planning are performed on the level of macro actions, and vehicles execute macro actions on the surface level.

The task of macro actions is to parametrise, initialise, and chain maneuvers together, to track the state of these maneuvers, and to advance to new maneuvers once the current ones have terminated.

To create a new macro action, you should create a new class which inherits from the class `igp2.planlibrary.macro_action.MacroAction`.
The parent class `MacroAction` defines three main methods that should be overwritten before the macro action can be used.

The following code snippet describes these methods.

```python
import igp2 as ip

class NewMacroAction(ip.MacroAction):
    """ My new macro action. """

    def get_maneuvers(self) -> List[Maneuver]:
        """ The most crucial method to override. This method is responsible
            for the initialisation, parametrisation, and chaining of maneuvers. """

    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ This method determines whether the macro action is applicable in the given 
            state of the environment. Failure to override this method will
            result in incorrectly setup macro actions in inapplicable states. """

    def get_possible_args(state: AgentState, scenario_map: Map, goal: Goal = None) \
            -> List[Dict]:
        """ Prediction and planning uses this method to generate all possible instances
            of the macro action in the given state. The list of dictionaries will be
            used to set up MacroActionConfigs """
```

Once you have created your new macro action, it is **essential** that you register it, otherwise planning and prediction will ignore it.

To register your new macro action, add the following lines to your script, before the IGP2 simulation has started.
```python
import igp2 as ip
type_str = "new_cl_maneuver"  # Enter a descriptive short name for your macro action here
type_macro = NewMacroAction  # The type of your macro action. I.e., the newly created class without parentheses.
ip.MacroActionFactory.register_new_macro(type_str, type_macro)  # Register your new macro action with IGP2
```

### What's next?

After all these steps you can finally run your simulation with the newly added behaviour.

During its running, IGP2 refers to the registered macro actions and maneuvers in the `MacroActionFactory` and `CLManeuverFactory` classes to fetch all applicable macro actions.

After having registered your new actions, IGP2 will start planning with them, and they also become available to use in configuration files.

### What if my new code doesn't seem to be working?

IGP2 is a complex piece of software so there is likely that some bugs have crept into your code.

Here are a few suggestions and steps to take when hunting for bugs:
1. Plot your macro action in prediction: You can set `debug=True` when calling the A* search during goal recognition (see, `igp2.recognition.astar.AStar.search()). This will plot every iteration of A* and how your trajectories are being generated during the search.
2. Similarly, you can plot the rollouts of MCTS by setting `plot_rollout=True` in the call to `igp2.planning.rollout.run()`.
3. Make sure that you look out for edge cases. Errors can happen when the vehicle is very close to the end of its current lane.
4. Similarly, try to avoid setting the velocity of the vehicle to zero. If you need to simulate stopping, then set velocity to some very small non-zero value. 

[Next: Overwriting MCTS](overwrite_mcts.md)

