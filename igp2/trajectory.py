import numpy as np
from typing import Union, Tuple, List, Dict


from typing import List

from igp2.agent import AgentState


class Trajectory:
    # TODO: Finish implementing all functionality
    def __init__(self, fps: int, start_time: float, frames: List[AgentState] = None):
        self.fps = fps
        self.start_time = start_time
        self._state_list = frames if frames is not None else []

    @property
    def states(self):
        return self._state_list

    @property
    def length(self) -> float:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError

    def add_state(self, new_state: AgentState):
        # TODO: Add sanity checks
        self._state_list.append(new_state)


class VelocityTrajectory:
    """ Define a trajectory consisting of a 2d path and velocities """

    def __init__(self, path, velocity):
        """ Create a VelocityTrajectory object

        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        self.path = path
        self.velocity = velocity

