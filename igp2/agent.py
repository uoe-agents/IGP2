from dataclasses import dataclass
from typing import Union, Tuple, Dict

import numpy as np
import abc

from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map


@dataclass
class AgentState:
    """ Dataclass storing data points that describe an exact moment in a trajectory.

     The time field may represent either continuous time or time steps. Velocity and acceleration can be
     given both with vectors or floats.
     """
    time: float
    position: np.ndarray
    velocity: Union[float, np.ndarray]
    acceleration: Union[float, np.ndarray]
    heading: float
    lane: Lane = None
    frame_of_closest_agent: "AgentState" = None

    def __copy__(self):
        return AgentState(self.time, self.position, self.velocity, self.acceleration,
                          self.heading, lane=self.lane, frame_of_closest_agent=self.frame_of_closest_agent)

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    def to_hashable(self) -> Tuple[float, float, float, float, float]:
        """ Returns a hashable representation of the state, that is useful for MCTS.

        Returns:
            5-tuple of the form (x-pos, y-pos, speed, heading, time)
        """
        return self.position[0], self.position[1], self.speed, self.heading, self.time
    

@dataclass(eq=True, frozen=True)
class AgentMetadata:
    """ Represents the physical properties of the Agent. """
    agent_id: int
    width: float
    length: float
    agent_type: str
    initial_time: float
    final_time: float

    @classmethod
    def default_meta(cls, frame: Dict[int, AgentState]) -> Dict[int, "AgentMetadata"]:
        """ Create a dictionary of metadata for agents in the given frame using the default agent metadata"""
        return {
            aid: cls(aid, 1.8, 4.6, "car", state.time, None) for aid, state in frame.items()
        }


class Agent(abc.ABC):
    """ Abstract class for all agents. """

    def __init__(self, agent_id: int, agent_metadata: AgentMetadata, goal: "Goal" = None):
        """ Initialise base fields of the agent.

        Args:
            agent_id: ID of the agent
            agent_metadata: metadata describing the properties of the agent
            goal: optional final goal of the agent
        """
        self._agent_id = agent_id
        self._metadata = agent_metadata
        self._goal = goal

    def done(self, frame: Dict[int, AgentState], scenario_map: Map) -> bool:
        raise NotImplementedError()

    def next_action(self, frame: Dict[int, AgentState], scenario_map: Map):
        raise NotImplementedError()

    @property
    def agent_id(self) -> int:
        """ ID of the agent"""
        return self._agent_id

    @property
    def metadata(self) -> AgentMetadata:
        """ Metadata describing the physical properties of the agent"""
        return self._metadata

    @property
    def goal(self) -> "Goal":
        """ Final goal of the agent"""
        return self._goal


class TrajectoryAgent(Agent):
    """ Agent that follows a predefined trajectory. """

    def __init__(self,
                 agent_id: int,
                 agent_metadata: AgentMetadata,
                 goal: "Goal" = None,
                 trajectory: "Trajectory" = None):
        """ Initialise new trajectory-following agent.

        Args:
            agent_id: ID of the agent
            agent_metadata: Metadata describing the properties of the agent
            trajectory: optional initial trajectory
        """
        super().__init__(agent_id, agent_metadata, goal)
        self._trajectory = trajectory

    def done(self, frame: Dict[int, AgentState], scenario_map: Map) -> bool:
        pass

    def next_action(self, frame: Dict[int, AgentState], scenario_map: Map):
        pass

    @property
    def trajectory(self) -> "Trajectory":
        """ Return the currently defined trajectory of the agent. """
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value: "Trajectory"):
        """ Overwrite current trajectory of agent with value"""
        self._trajectory = value


class MacroAgent(Agent):
    """ Agent executing a pre-defined macro action. Useful for simulating the ego vehicle during MCTS. """

    def __init__(self, agent_id: int, agent_metadata: AgentMetadata,
                 goal: "Goal" = None, macro_action: "MacroAction" = None):
        """ Create a new macro agent. """
        super().__init__(agent_id, agent_metadata, goal)
        self._current_macro = macro_action
        self._current_maneuver = None  # TODO

    def done(self, frame: Dict[int, AgentState], scenario_map: Map) -> bool:
        raise NotImplementedError

    def next_action(self, frame: Dict[int, AgentState], scenario_map: Map):
        raise NotImplementedError

    def update_macro_action(self, new_macro_action: "MacroAction"):
        """ Overwrite current macro action of the agent.

        Args:
            new_macro_action: new macro action to execute
        """
        self._current_macro = new_macro_action
