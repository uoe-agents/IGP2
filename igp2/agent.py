from dataclasses import dataclass
import numpy as np
import abc

from igp2.opendrive.elements.road_lanes import Lane


@dataclass
class AgentState:
    """ Dataclass storing data points that describe an exact moment in a trajectory.

     The time field may represent either continuous time or time steps.
     """
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float
    max_speed: float
    lane: Lane = None
    frame_of_closest_agent = None

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)
    

@dataclass
class AgentMetadata:
    """ Represents the physical properties of the Agent. """
    agent_id: int
    width: float
    length: float
    agent_type: str
    initial_time: float
    final_time: float


class Agent(abc.ABC):
    def __init__(self, agent_id: int, agent_metadata: AgentMetadata, view_radius: float = None):
        self.agent_id = agent_id
        self.metadata = agent_metadata
        self.view_radius = view_radius

    def done(self) -> bool:
        raise NotImplementedError()

    def next_action(self, observation: "Observation" = None):
        raise NotImplementedError()


class TrajectoryAgent(Agent):
    def __init__(self, agent_id: int, agent_metadata: AgentMetadata, trajectory: "Trajectory"):
        super().__init__(agent_id, agent_metadata)
        self.trajectory = trajectory

    def done(self) -> bool:
        raise NotImplementedError()

    def next_action(self, observation: "Observation" = None):
        raise NotImplementedError()
