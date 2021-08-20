from dataclasses import dataclass
from typing import Union, Tuple, Dict

import numpy as np

from igp2.opendrive.elements.road_lanes import Lane


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