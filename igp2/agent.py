from typing import Dict

import abc

from igp2.agentstate import AgentState, AgentMetadata
from igp2.goal import Goal
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.trajectory import Trajectory
from igp2.vehicle import Vehicle, Observation


class Agent(abc.ABC):
    """ Abstract class for all agents. """

    def __init__(self,
                 agent_id: int,
                 state: AgentState,
                 metadata: AgentMetadata,
                 goal: "Goal" = None):
        """ Initialise base fields of the agent.

        Args:
            agent_id: ID of the agent
            state: Starting state of the agent
            metadata: Metadata describing the properties of the agent
            goal: Optional final goal of the agent
        """
        self._agent_id = agent_id
        self._state = state
        self._metadata = metadata
        self._goal = goal

        self._vehicle = Vehicle(state, metadata)

    def done(self, observation: Observation) -> bool:
        """ Check whether the agent has completed executing its assigned task. """
        raise NotImplementedError

    def next_action(self, observation: Observation):
        """ Return the next action the agent will take. """
        raise NotImplementedError

    def update_goal(self, new_goal: "Goal"):
        """ Overwrite the current goal of the agent"""
        self._goal = new_goal

    @property
    def agent_id(self) -> int:
        """ ID of the agent"""
        return self._agent_id

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def metadata(self) -> AgentMetadata:
        """ Metadata describing the physical properties of the agent"""
        return self._metadata

    @property
    def goal(self) -> "Goal":
        """ Final goal of the agent"""
        return self._goal

    @property
    def vehicle(self) -> "Vehicle":
        """ Return the physical vehicle attached to this agent. """
        return self._vehicle


class TrajectoryAgent(Agent):
    """ Agent that follows a predefined trajectory. """

    def __init__(self,
                 agent_id: int,
                 state: AgentState,
                 metadata: AgentMetadata,
                 goal: "Goal" = None,
                 trajectory: "Trajectory" = None):
        """ Initialise new trajectory-following agent.

        Args:
            agent_id: ID of the agent
            state: Starting state of the agent
            metadata: Metadata describing the properties of the agent
            trajectory: Optional initial trajectory
        """
        super().__init__(agent_id, state, metadata, goal)
        self._trajectory = trajectory

    def done(self, observation: Observation) -> bool:
        raise NotImplementedError

    def next_action(self, observation: Observation):
        raise NotImplementedError

    @property
    def trajectory(self) -> Trajectory:
        """ Return the currently defined trajectory of the agent. """
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value: Trajectory):
        """ Overwrite current trajectory of agent with value"""
        self._trajectory = value


class MacroAgent(Agent):
    """ Agent executing a pre-defined macro action. Useful for simulating the ego vehicle during MCTS. """

    def __init__(self,
                 agent_id: int,
                 state: AgentState,
                 metadata: AgentMetadata,
                 goal: Goal = None):
        """ Create a new macro agent. """
        super().__init__(agent_id, state, metadata, goal)

        self._current_macro = None
        self._current_maneuver = None  # TODO

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
        return self.vehicle.get_state(observation.frame[self.agent_id].time)

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
        for args in new_macro_action.get_possible_args(frame[self.agent_id], scenario_map, self._goal.center):
            self._current_macro = new_macro_action(agent_id=self.agent_id,
                                                   frame=frame,
                                                   scenario_map=scenario_map,
                                                   open_loop=False,
                                                   **args)
