import abc
import numpy as np
import logging
from typing import Union, Optional
from typing import List

from igp2.core.agentstate import AgentState
from igp2.core.util import get_curvature

logger = logging.getLogger(__name__)


class Trajectory(abc.ABC):
    """ Base class for all Trajectory objects """

    VELOCITY_STOP = 0.1

    def __init__(self, path: np.ndarray = None, velocity: np.ndarray = None):
        """ Create an empty Trajectory object

        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        self._path = path
        self._velocity = velocity
        self._velocity_stop = Trajectory.VELOCITY_STOP

    def __len__(self):
        return len(self._path)

    @property
    def path(self) -> np.ndarray:
        """ Sequence of positions along a path. """
        return self._path if self._path is not None else np.array([])

    @property
    def velocity(self) -> np.ndarray:
        """ Velocities corresponding to each position along the path. """
        return self._velocity if self._velocity is not None else np.array([])

    @velocity.setter
    def velocity(self, array: np.ndarray = None):
        self._velocity = array if array is not None else np.array([])

    @property
    def acceleration(self) -> np.ndarray:
        """ Accelerations corresponding to each position along the path. """
        var = self.differentiate(self.velocity, self.times)
        var = np.where(abs(var) >= 1e2, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def jerk(self) -> np.ndarray:
        """ Jerk values corresponding to each position along the path. """
        var = self.differentiate(self.acceleration, self.times)
        var = np.where(abs(var) >= 1e3, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def angular_velocity(self) -> np.ndarray:
        """ Calculates angular velocity (positive counter-clockwise), handling discontinuity at theta = pi """
        var = self.differentiate(np.unwrap(self.heading), self.times)
        var = np.where(abs(var) >= 1e1, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def angular_acceleration(self) -> np.ndarray:
        """ Calculates angular acceleration (positive counter-clockwise). """
        var = self.differentiate(self.angular_velocity, self.times)
        var = np.where(abs(var) >= 1e3, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def curvature(self) -> np.ndarray:
        """ Calculates curvature of the trajectory at each point in the path. """
        var = np.nan_to_num(get_curvature(self.path), posinf=0.0, neginf=0.0)
        var = np.where(abs(var) >= 1e3, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def velocity_stop(self) -> float:
        """ Velocity at or under which the vehicle is considered to be at a stop """
        return self._velocity_stop

    @property
    def initial_agent_state(self) -> AgentState:
        """ AgentState calculated from the first point along the path. """
        return AgentState(0, self.path[0], self.velocity[0], self.acceleration[0], self.heading[0])

    @property
    def final_agent_state(self) -> AgentState:
        """ AgentState calculated from the final point along the path. """
        return AgentState(0, self.path[-1], self.velocity[-1], self.acceleration[-1], self.heading[-1])

    @property
    def length(self) -> Optional[float]:
        """ Length of trajectory in metres or None if trajectory is empty. """
        raise NotImplementedError

    @property
    def timesteps(self) -> Optional[np.ndarray]:
        """ Time elapsed between two consecutive trajectory points in seconds. """
        raise NotImplementedError

    @property
    def times(self) -> Optional[np.ndarray]:
        """ Time elapsed (from start) along the trajectory in seconds. """
        if self.timesteps is not None and len(self.timesteps) > 0:
            return np.cumsum(self.timesteps)
        else:
            return None

    @property
    def duration(self) -> Optional[float]:
        """ Duration in seconds to cover the path with given velocities. """
        if self.times is not None and len(self.times) > 0:
            return self.times[-1]
        else:
            return None

    @property
    def heading(self) -> np.ndarray:
        """ Heading of vehicle in radians.
        
        Returns:
             array of heading value from dataset, or estimate from trajectory 
             path if heading data is not available
        """
        raise NotImplementedError

    def differentiate(self, x: np.ndarray, y: np.ndarray, dx: np.ndarray = None, dy: np.ndarray = None):
        """ Performs backward difference on data x y.

        Notes:
            - First element is computed by forward difference using a continuity assumption.
            - Can overload dx and dy if required.
            - Will replace nonsensical values (Nan and +/- inf with 0.0).
        """
        if dx is None: dx = np.diff(x, axis=0)
        if dy is None: dy = np.diff(y, axis=0)
        dx_dy = np.divide(dx, dy)
        dx_dy = np.insert(dx_dy, 0, dx_dy[0])
        return np.nan_to_num(dx_dy, posinf=0.0, neginf=0.0)

    def heading_from_path(self, path) -> np.ndarray:
        """ Calculate headings at each point along the trajectory. """
        dpath = np.diff(path, axis=0)
        heading = np.angle(dpath.view(dtype=np.complex128)).reshape(-1)
        heading = np.insert(heading, 0, heading[0])
        return heading

    def trajectory_dt(self, path, velocity):
        """ Calculate time elapsed between two consecutive points along the trajectory
        using the mean of the two velocities."""
        # assume constant acceleration between points on path
        v_avg = (velocity[:-1] + velocity[1:]) / 2
        s = np.linalg.norm(np.diff(path, axis=0), axis=1)
        dt = np.concatenate([[0], s / v_avg])
        return dt

    def extend(self, new_trajectory: "Trajectory"):
        """ Extend the trajectory in-place."""
        raise NotImplementedError

    def slice(self, start_idx: Optional[int], end_idx: Optional[int]) -> "Trajectory":
        """ Return a slice of the trajectory between the given indices. Follows regular Python indexing standards. """
        raise NotImplementedError


class StateTrajectory(Trajectory):
    """ Implements a Trajectory that is built from discreet observations at given time intervals. """

    def __init__(self, fps: int, states: List[AgentState] = None,
                 path: np.ndarray = None, velocity: np.ndarray = None):
        """ Create a new StateTrajectory. Path and velocity fields are populated from the given frames.

        Args:
            fps: The number of time steps each second
            states: Optionally, specify a list of AgentStates with an observed trajectory
            path: Path points along the trajectory. Ignored.
            velocity: Velocities at each path point. Ignored.
        """
        super().__init__(path, velocity)
        self.fps = fps
        self._state_list = states if states is not None else []
        self.calculate_path_and_velocity()

    def __getitem__(self, item: int) -> AgentState:
        return self.states[item]

    def __iter__(self):
        yield from self.states

    def __len__(self):
        return len(self.states)

    @classmethod
    def from_velocity_trajectory(cls,
                                 velocity_trajectory: "VelocityTrajectory",
                                 fps: int = 20) -> "StateTrajectory":
        """ Convert a velocity trajectory to a StateTrajectory.

        Args:
            velocity_trajectory: VelocityTrajectory to convert
            fps: Optional framerate argument
        """
        states = []

        # Interpolate trajectory to match FPS
        num_frames = int(np.ceil(velocity_trajectory.duration * fps))
        ts = velocity_trajectory.times
        points = np.linspace(ts[0], ts[-1], int(num_frames))

        xs_r = np.interp(points, ts, velocity_trajectory.path[:, 0])
        ys_r = np.interp(points, ts, velocity_trajectory.path[:, 1])
        v_r = np.interp(points, ts, velocity_trajectory.velocity)
        a_r = np.interp(points, ts, velocity_trajectory.acceleration)
        h_r = np.interp(points, ts, velocity_trajectory.heading)
        path = np.c_[xs_r, ys_r]

        for i in range(num_frames):
            states.append(AgentState(time=i,
                                     position=path[i],
                                     velocity=v_r[i],
                                     acceleration=a_r[i],
                                     heading=h_r[i]))
        trajectory = cls(fps, states=states, path=path, velocity=v_r)
        return trajectory

    @property
    def states(self) -> List[AgentState]:
        """ Return the list of states. """
        return self._state_list

    @property
    def length(self) -> Optional[float]:
        if self._path is not None:
            return np.linalg.norm(np.diff(self._path, axis=0), axis=1).sum()
        else:
            return None

    @property
    def timesteps(self) -> Optional[np.ndarray]:
        if self.fps is not None and self.fps > 0.0:
            timesteps = np.array([1 / self.fps] * len(self._state_list))
            timesteps[0] = 0
            return timesteps
        elif self.path is not None and len(self.path) == 1:
            return np.zeros(1)
        elif self.path is not None and len(self.path) > 1:
            return self.trajectory_dt(self.path, self.velocity)
        else:
            return None

    @property
    def heading(self) -> Optional[np.ndarray]:
        if self._state_list:
            heading = [state.heading for state in self._state_list]
            return np.array(heading)
        else:
            return self.heading_from_path(self.path)

    def calculate_path_and_velocity(self):
        """ Recalculate path and velocity fields. May be used when the trajectory is updated. """
        if self._state_list and len(self._state_list) > 0:
            self._path = np.array([state.position for state in self._state_list])
            self._velocity = np.array([state.speed for state in self._state_list])

    def add_state(self, new_state: AgentState, reload_path: bool = True):
        """ Add a new state at the end of the trajectory.

        Args:
            new_state: AgentState. This should follow the last state of the trajectory in time.
            reload_path: If True then the path and velocity fields are recalculated.
        """
        self._state_list.append(new_state)

        if reload_path:
            if self._path is None or self._velocity is None:
                self.calculate_path_and_velocity()
            else:
                self._path = np.append(self._path, np.array([new_state.position]), axis=0)
                self._velocity = np.append(self._velocity, new_state.speed)

    def extend(self, new_trajectory: "StateTrajectory", reload_path: bool = True, reset_times: bool = False):
        """ Extend the current trajectory with the states of the given trajectory. If the last state of the first
         trajectory is equal to the first state of the second trajectory then the first state of the second trajectory
         is dropped.

        Args:
            new_trajectory: The given trajectory to use for extension.
            reload_path: Whether to recalculate the path and velocity fields.
            reset_times: Whether to reset the timing information on the states.
        """
        start_idx = 0
        if len(self.states) > 0 and np.allclose(self.states[-1].position, new_trajectory.states[0].position):
            start_idx = 1

        if reset_times:
            start_time = self._state_list[-1].time
            for i, state in enumerate(new_trajectory.states[start_idx:], 1):
                state.time = start_time + i

        self._state_list.extend(new_trajectory.states[start_idx:])

        if reload_path:
            self.calculate_path_and_velocity()

    def slice(self, start_idx: Optional[int], end_idx: Optional[int]) -> "StateTrajectory":
        """ Return a slice of the original StateTrajectory"""
        return StateTrajectory(self.fps,
                               self._state_list[start_idx:end_idx],
                               self.path[start_idx:end_idx],
                               self.velocity[start_idx:end_idx])


class VelocityTrajectory(Trajectory):
    """ Define a trajectory consisting of a 2d paths and corresponding velocities """

    def __init__(self, path: np.ndarray, velocity: np.ndarray, heading: np.ndarray = None,
                 timesteps: np.ndarray = None):
        """ Create a VelocityTrajectory object

        Args:
            path: nx2 array containing sequence of points.
            velocity: array containing velocity at each point.
            heading: optional array containing heading (radions) at each point.
            timesteps: optional array containing the time elapsed between each consecutive point.
        """
        super().__init__(path, velocity)
        self._pathlength = self.calculate_pathlength(path)

        if heading is None:
            self._heading = self.heading_from_path(self.path)
        else:
            self._heading = heading

        if timesteps is None:
            if len(self.path) == 1:
                self._timesteps = np.zeros(1)
            else:
                self._timesteps = self.trajectory_dt(self.path, self.velocity)
        else:
            self._timesteps = timesteps

    @property
    def pathlength(self) -> np.ndarray:
        """ Length of path travelled at each position along the path in meters. """
        return self._pathlength if self._pathlength is not None else np.array([])

    @property
    def length(self) -> float:
        return self._pathlength[-1]

    @property
    def heading(self) -> np.ndarray:
        return self._heading

    @property
    def timesteps(self) -> Optional[np.ndarray]:
        return self._timesteps

    @classmethod
    def from_agent_state(cls, state: AgentState) -> "VelocityTrajectory":
        heading = np.array([state.heading]) if state.heading is not None else None
        return cls(np.array([state.position]),
                   np.array([state.speed]),
                   heading)

    @classmethod
    def from_agent_states(cls, states: List[AgentState]) -> "VelocityTrajectory":
        heading = np.array([s.heading for s in states])
        return cls(np.array([s.position for s in states]),
                   np.array([s.speed for s in states]),
                   heading)

    def calculate_pathlength(self, path) -> np.ndarray:
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)  # Length between points
        return np.cumsum(np.append(0, path_lengths))

    def insert(self, trajectory: Trajectory):
        """ Inserts a Trajectory (in-place) at the beginning of the VelocityTrajectory object,
            removing the first element of the original VelocityTrajectory. """
        if trajectory.velocity.size == 0:
            pass
        else:
            path = np.concatenate((trajectory.path, self.path[1:]))
            velocity = np.concatenate((trajectory.velocity, self.velocity[1:]))
            heading = np.concatenate((trajectory.heading, self.heading[1:]))
            timesteps = np.concatenate((trajectory.timesteps, self.timesteps[1:]))
            self.__init__(path, velocity, heading, timesteps)

    def extend(self, new_trajectory: Union[np.ndarray, Trajectory]):
        """ Extends a Trajectory (in-place) at the end of the VelocityTrajectory object. """
        if isinstance(new_trajectory, Trajectory):
            path_p1 = np.concatenate([[self.path[-1]], new_trajectory.path], axis=0)
            vel_p1 = np.concatenate([[self.velocity[-1]], new_trajectory.velocity])
            extra_timestep = self.trajectory_dt(path=path_p1[0:2], velocity=vel_p1[0:2])[-1]
            self._path = np.concatenate([self.path, new_trajectory.path], axis=0)
            self._velocity = np.concatenate([self.velocity, new_trajectory.velocity])
            self._timesteps = np.concatenate([self.timesteps, [extra_timestep], new_trajectory.timesteps[1:]])
        else:
            path_p1 = np.concatenate([[self.path[-1]], new_trajectory[0]], axis=0)
            vel_p1 = np.concatenate([[self.velocity[-1]], new_trajectory[1]])
            timesteps = self.trajectory_dt(path_p1, vel_p1)
            self._path = np.concatenate([self.path, new_trajectory[0]], axis=0)
            self._velocity = np.concatenate([self.velocity, new_trajectory[1]])
            self._timesteps = np.concatenate([self.timesteps, timesteps[1:]])
        heading = self.heading_from_path(path_p1)
        self._heading = np.concatenate([self.heading, heading[1:]])
        self._pathlength = self.calculate_pathlength(self._path)

    def slice(self, start_idx: Optional[int], end_idx: Optional[int]) -> "VelocityTrajectory":
        """ Return a slice of the original VelocityTrajectory"""
        return VelocityTrajectory(self.path[start_idx:end_idx],
                                  self.velocity[start_idx:end_idx],
                                  self._heading[start_idx:end_idx],
                                  self._timesteps[start_idx:end_idx])
