from dataclasses import dataclass, field
from typing import Union, Tuple, Dict

import numpy as np

from igp2.opendrive.elements.road_lanes import Lane


@dataclass
class AgentMetadata:
    """ Represents the physical properties of the Agent. """

    # For a Skoda Octavia IV 2.0 TDI (150 Hp) DSG
    # Ref: https://www.auto-data.net/en/skoda-octavia-iv-2.0-tdi-150hp-dsg-38011
    CAR_DEFAULT = {
        "width": 1.829,
        "length": 4.689,
        "height": 1.47,
        "agent_type": "car",
        "wheelbase": 2.686,
        "front_overhang": 0.91,
        "rear_overhang": 1.094,
        "front_track": 1.543,
        "back_track": 1.535,
        "drag_coefficient": 0.252,
        "max_acceleration": 5.0,  # this is not to true specs but to be consistent with constraints placed on velocity smoother.
    }

    # TODO: Add truck/bus default

    width: float
    length: float
    agent_type: str
    initial_time: float = None
    final_time: float = None
    height: float = None
    wheelbase: float = None  # Distance between axles
    front_overhang: float = None
    rear_overhang: float = None
    front_track: float = None  # Distance between front wheels
    back_track: float = None
    drag_coefficient: float = None
    max_acceleration: float = None

    @classmethod
    def default_meta_frame(cls, frame: Dict[int, "AgentState"]) -> Dict[int, "AgentMetadata"]:
        """ Create a dictionary of metadata for agents in the given frame using the default agent metadata"""
        ret = {}
        for aid, state in frame.items():
            ret[aid] = cls(initial_time=state.time, **AgentMetadata.CAR_DEFAULT)
        return ret

    @classmethod
    def interleave(cls, meta_dest: "AgentMetadata", meta_src: Union[dict, "AgentMetadata"]) -> "AgentMetadata":
        for field in cls.__dict__.keys():
            if getattr(meta_dest, field) is None:
                value = None
                if isinstance(meta_src, dict) and field in meta_src:
                    value = meta_src[field]
                elif isinstance(meta_src, type(cls)):
                    value = getattr(meta_src, field)

                if value is not None:
                    setattr(meta_dest, field, value)
        return meta_dest


@dataclass
class AgentState:
    """ Dataclass storing data points that describe an exact moment in a trajectory.

     The time field may represent either continuous time or time steps. Velocity and acceleration can be
     given both with vectors or floats.
     """
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float
    metadata: AgentMetadata = field(default_factory=lambda: AgentMetadata(**AgentMetadata.CAR_DEFAULT))

    def __copy__(self):
        return AgentState(self.time, self.position, self.velocity, self.acceleration, self.heading, self.metadata)

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    def to_hashable(self) -> Tuple[float, float, float, float, float]:
        """ Returns a hashable representation of the state, that is useful for MCTS.

        Returns:
            5-tuple of the form (x-pos, y-pos, speed, heading, time)
        """
        return self.position[0], self.position[1], self.speed, self.heading, self.time
