import abc
import numpy as np
import casadi as ca
from typing import Union, Tuple, List, Dict

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

class VelocitySmoother(VelocityTrajectory):

    def __init__(self, path, velocity, n: int = 100, dt_s: float = 0.1,
     amax_m_s2: float = 5.0, vmax_m_s: float = 10.0, lambda_acc: float = 10.0):
        super().__init__(path, velocity)
        self._n = n
        self._dt = dt_s
        self._amax = amax_m_s2
        self._vmax = vmax_m_s
        self._lambda_acc = lambda_acc

        self._pathlength = self._curvelength(self.path)

    def _curvelength(self, path):
        path_lengths = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)) # Length between points
        path_lengths_inc = [0] * len(path)
        for i in range(1, len(path_lengths)+1):
            path_lengths_inc[i] = path_lengths[i-1] + path_lengths_inc[i-1]
        return np.array(path_lengths_inc)

    def lin_interpolant(self, x, y):
        """Creates a differentiable Casadi interpolant object to linearly interpolate y from x"""
        #TODO overload function to use indices when x is not provided
        return ca.interpolant('LUT', 'linear', [x], y)

    def smooth_velocity(self, debug: bool = False):
        #TODO: recursively run optimisation if x[n] > pathlength[-1]
        return self.optimiser(debug)

    def optimiser(self, debug: bool = False):
        #TODO: remove printouts to terminal when debug is false
        opti = ca.Opti()

        # Create optimisation variables
        x = opti.variable(self.n)
        v = opti.variable(self.n)
        acc = opti.variable(self.n - 1)
        for k in range(0, self.n - 1):
            acc[k] = v[k+1] - v[k]

        # Create interpolants for pathlength and velocity
        ind = [i for i in range(len(self.pathlength))]
        pathlength_interpolant = self.lin_interpolant(ind, self.pathlength)
        velocity_interpolant = self.lin_interpolant(self.pathlength, self.velocity)

        # Initialise optimisation variables
        ind_n = [i * len(self.pathlength) / (self.n - 1) for i in range(self.n)] 
        path_ini = pathlength_interpolant(ind_n)
        vel_ini = velocity_interpolant(path_ini)
        opti.set_initial(x, path_ini)
        opti.set_initial(v, vel_ini)

        # Optimisation objective to minimise
        J = ca.sumsqr(v - velocity_interpolant(x)) + self.lambda_acc * ca.sumsqr(acc)
        opti.minimize( J )

        # Optimisation constraints
        for k in range(0 , self.n - 1):
            opti.subject_to( x[k+1] == x[k] + v[k] * self.dt )
            opti.subject_to( ca.sqrt((v[k + 1] - v[k])**2) < self.amax * self.dt )

        opti.subject_to( x[0] == self.pathlength[0])
        opti.subject_to( v[0] == self.velocity[0])

        opti.subject_to( opti.bounded(0, v, self.vmax))
        opti.subject_to(v <= velocity_interpolant(x))

        opti.solver('ipopt')

        sol = opti.solve()

        sol_interpolant = self.lin_interpolant(sol.value(x), sol.value(v))
        
        return sol_interpolant(self.pathlength)

    @property
    def n(self) -> int:
        """Returns the number of points used in smoothing optimisation"""
        return self._n

    @property
    def dt(self) -> float:
        """Returns the timestep size used in smoothing optimisation in seconds"""
        return self._dt

    @property
    def amax(self) -> float:
        """Returns the acceleration limit used in smoothing optimisation in m/s2"""
        return self._amax

    @property
    def vmax(self) -> float:
        """Returns the velocity limit used in smoothing optimisation in m/s"""
        return self._vmax

    @property
    def lambda_acc(self) -> float:
        """Returns the lambda parameter used to control the weighting of the 
        acceleration penalty in smoothing optimisation"""
        return self._lambda_acc

    @property
    def pathlength(self):
        """Returns the cummulative length travelled in the trajectory in meters"""
        return self._pathlength
