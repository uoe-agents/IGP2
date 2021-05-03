import abc
import numpy as np

from typing import List

from igp2.agent import AgentState


class Trajectory(abc.ABC):
    """ Base class for all Trajectory objects """
    def __init__(self, path: np.ndarray = None, velocity: np.ndarray = None):
        """ Create an empty Trajectory object

        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        self._path = path
        self._velocity = velocity

    @property
    def path(self) -> np.ndarray:
        """ Sequence of positions along a path. """
        return self._path if self._path is not None else np.array([])

    @property
    def velocity(self) -> np.ndarray:
        """ Velocities corresponding to each position along the path. """
        return self._velocity if self._velocity is not None else np.array([])

    @property
    def length(self) -> float:
        """ Length of the path in metres. """
        raise NotImplementedError

    @property
    def duration(self) -> float:
        """ Duration in seconds to cover the path with given velocities. """
        raise NotImplementedError


class StateTrajectory(Trajectory):
    """ Implements a Trajectory that is build discreet observations at each time step. """
    def __init__(self, fps: int, start_time: int, frames: List[AgentState] = None,
                 path: np.ndarray = None, velocity: np.ndarray = None):
        """ Create a new StateTrajectory

        Args:
            fps: The number of time steps each second
            start_time: The first time step of the StateTrajectory
            frame: Optionally, specify a list of AgentStates with an observed trajectory
            path: Path points along the trajectory. Ignored.
            velocity: Velocities at each path point. Ignored.
        """
        super().__init__(path, velocity)
        self.fps = fps
        self.start_time = start_time
        self._state_list = frames if frames is not None else []
        self._calculate_path_velocity()

    def __getitem__(self, item: int) -> AgentState:
        return self._state_list[item]

    def __iter__(self):
        yield from self._state_list

    @property
    def path(self) -> np.ndarray:
        return self._path

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def states(self) -> List[AgentState]:
        return self._state_list

    @property
    def length(self) -> float:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError

    def _calculate_path_velocity(self):
        if self._state_list is not None:
            self._path = np.array([state.position for state in self._state_list])
        if self._state_list is not None:
            self._velocity = np.array([state.velocity for state in self._state_list])

    def add_state(self, new_state: AgentState):
        """ Add a new state at the end of the trajectory

        Args:
            new_state: AgentState. This should follow the last state of the trajectory in time.
        """
        if len(self._state_list) > 0:
            assert self._state_list[-1].time < new_state.time
        self._state_list.append(new_state)
        self._path = np.append(self._path, new_state.position)
        self._velocity = np.append(self._path, new_state.speed)

    def extend(self, trajectory):
        """ Extend the current trajectory with """
        if len(self._state_list) > 0:
            assert self._state_list[-1].time < trajectory.states[0].time
        self._state_list.extend(trajectory.states)
        self._calculate_path_velocity()


class VelocityTrajectory(Trajectory):
    """ Define a trajectory consisting of a 2d path and velocities """
    @property
    def length(self) -> float:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError
