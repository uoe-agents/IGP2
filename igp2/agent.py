from typing import Dict
import numpy as np
import abc

from shapely.geometry import Point

from igp2.agentstate import AgentState, AgentMetadata
from igp2.goal import Goal
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction, Exit
from igp2.trajectory import Trajectory, StateTrajectory, VelocityTrajectory
from igp2.vehicle import Vehicle, Observation, TrajectoryVehicle, KinematicVehicle, Action


class Agent(abc.ABC):
    """ Abstract class for all agents. """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 metadata: AgentMetadata,
                 goal: "Goal" = None,
                 fps: int = 20):
        """ Initialise base fields of the agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            metadata: Metadata describing the properties of the agent
            goal: Optional final goal of the agent
            fps: Execution rate of the environment simulation
        """
        self._alive = True

        self._agent_id = agent_id
        self._metadata = metadata
        self._initial_state = initial_state
        self._goal = goal
        self._fps = fps
        self._vehicle = None

    def done(self, observation: Observation) -> bool:
        """ Check whether the agent has completed executing its assigned task. """
        raise NotImplementedError

    def next_action(self, observation: Observation):
        """ Return the next action the agent will take. """
        raise NotImplementedError

    def update_goal(self, new_goal: "Goal"):
        """ Overwrite the current goal of the agent. """
        self._goal = new_goal

    def reset(self):
        """ Reset agent to initialisation defaults. """
        self._alive = True
        self._vehicle = None

    @property
    def agent_id(self) -> int:
        """ ID of the agent. """
        return self._agent_id

    @property
    def state(self) -> AgentState:
        """ Return current state of the agent as given by its vehicle. """
        return self._vehicle.get_state()

    @property
    def metadata(self) -> AgentMetadata:
        """ Metadata describing the physical properties of the agent. """
        return self._metadata

    @property
    def goal(self) -> "Goal":
        """ Final goal of the agent. """
        return self._goal

    @property
    def vehicle(self) -> "Vehicle":
        """ Return the physical vehicle attached to this agent. """
        return self._vehicle

    @property
    def alive(self) -> bool:
        """ Whether the agent is alive in the simulation. """
        return self._alive

    @alive.setter
    def alive(self, value: bool):
        self._alive = value


class TrajectoryAgent(Agent):
    """ Agent that follows a predefined trajectory. """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 metadata: AgentMetadata,
                 goal: Goal = None,
                 fps: int = 20):
        """ Initialise new trajectory-following agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            metadata: Metadata describing the properties of the agent
            goal: Optional final goal of the vehicle
            fps: Execution rate of the environment simulation
        """
        super().__init__(agent_id, initial_state, metadata, goal, fps)

        self._vehicle = TrajectoryVehicle(initial_state, metadata, fps)
        self._t = 0
        self._trajectory = None

    def done(self, observation: Observation) -> bool:
        return self._t == len(self._trajectory.path) - 1

    def next_action(self, observation: Observation) -> AgentState:
        """ Calculate next action based on trajectory and set appropriate fields in vehicle. """
        assert self._trajectory is not None, f"Trajectory of Agent {self.agent_id} was None!"

        if self.done(observation):
            return self.state

        self._t += 1
        new_state = AgentState(
            self._t,
            self._trajectory.path[self._t],
            self._trajectory.velocity[self._t],
            self._trajectory.acceleration[self._t],
            self._trajectory.heading[self._t]
        )
        self._vehicle.execute_action(None, new_state)
        return new_state

    def set_trajectory(self, new_trajectory: Trajectory):
        """ Override current trajectory of the vehicle and resample to match execution frequency of the environment. """
        fps = self._vehicle.fps
        if isinstance(new_trajectory, StateTrajectory) and new_trajectory.fps == fps:
            self._trajectory = VelocityTrajectory(new_trajectory.path, new_trajectory.velocity)
        else:
            num_frames = np.ceil(new_trajectory.duration * fps)
            ts = new_trajectory.times
            points = np.linspace(ts[0], ts[-1], int(num_frames))

            xs_r = np.interp(points, ts, new_trajectory.path[:, 0])
            ys_r = np.interp(points, ts, new_trajectory.path[:, 1])
            v_r = np.interp(points, ts, new_trajectory.velocity)
            path = np.c_[xs_r, ys_r]
            self._trajectory = VelocityTrajectory(path, v_r)

    def reset(self):
        super(TrajectoryAgent, self).reset()
        self._vehicle = TrajectoryVehicle(self._initial_state, self._metadata, self._fps)
        self._t = 0
        self._trajectory = None

    @property
    def trajectory(self) -> Trajectory:
        """ Return the currently defined trajectory of the agent. """
        return self._trajectory


class MacroAgent(Agent):
    """ Agent executing a pre-defined macro action. Useful for simulating the ego vehicle during MCTS. """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 metadata: AgentMetadata,
                 goal: Goal = None,
                 fps: int = 20):
        """ Create a new macro agent. """
        super().__init__(agent_id, initial_state, metadata, goal, fps)
        self._vehicle = KinematicVehicle(initial_state, metadata, fps)
        self._current_macro = None

    def done(self, observation: Observation) -> bool:
        """ Returns true if the current macro action has reached a completion state. """
        assert self._current_macro is not None, f"Macro action of Agent {self.agent_id} is None!"
        return self._current_macro.done(observation.frame, observation.scenario_map)

    def next_action(self, observation: Observation) -> AgentState:
        """ Get the next action from the macro action and execute it through the attached vehicle of the agent.

        Args:
            observation: Observation of current environment state and road layout.

        Returns:
            The new state of the agent.
        """
        assert self._current_macro is not None, f"Macro action of Agent {self.agent_id} is None!"

        action = self._current_macro.next_action(observation)
        self.vehicle.execute_action(action)
        return self.vehicle.get_state(observation.frame[self.agent_id].time + 1)

    def reset(self):
        super(MacroAgent, self).reset()
        self._vehicle = KinematicVehicle(self._initial_state, self._metadata, self._fps)
        self._current_macro = None

    def update_macro_action(self,
                            new_macro_action: type(MacroAction),
                            observation: Observation):
        """ Overwrite and initialise current macro action of the agent. If multiple arguments are possible
        for the given macro, then choose the one that brings the agent closest to its goal.

        Args:
            new_macro_action: new macro action to execute
            observation: Current observation of the environment
        """
        frame = observation.frame
        scenario_map = observation.scenario_map
        possible_args = new_macro_action.get_possible_args(frame[self.agent_id], scenario_map, self._goal.center)

        # TODO: Possibly remove this check and consider each turn target separately in MCTS
        if len(possible_args) > 1 and isinstance(new_macro_action, type(Exit)):
            ps = np.array([t["turn_target"] for t in possible_args if "turn_target" in t])
            closest = np.argmin(np.linalg.norm(ps - self.goal.center, axis=1))
            possible_args = [{"turn_target": ps[closest]}]

        for args in possible_args:
            self._current_macro = new_macro_action(agent_id=self.agent_id,
                                                   frame=frame,
                                                   scenario_map=scenario_map,
                                                   open_loop=False,
                                                   **args)
