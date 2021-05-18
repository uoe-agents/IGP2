import abc
import numpy as np
import casadi as ca
from typing import Union, Tuple, List, Dict, Optional

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
    def length(self) -> Optional[float]:
        """ Length of the path in metres.

         Returns:
             length of trajectory in metres or None if trajectory is empty.
         """
        raise NotImplementedError

    @property
    def duration(self) -> Optional[float]:
        """ Duration in seconds to cover the path with given velocities.

        Returns:
             duration of trajectory in seconds or None if trajectory is empty.
        """
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
        self.calculate_path_velocity()

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
    def length(self) -> Optional[float]:
        if self._path is not None:
            return np.linalg.norm(np.diff(self._path, axis=0), axis=1).sum()
        else:
            return None

    @property
    def duration(self) -> Optional[float]:
        if self.fps is not None and self.fps > 0.0:
            return len(self._state_list) / self.fps
        elif self.path is not None and len(self.path) > 0:
            avg_velocities = (self.velocity[1:] + self.velocity[:-1]) / 2
            return (np.linalg.norm(np.diff(self.path, axis=0), axis=1) / avg_velocities).sum()
        else:
            return None

    def calculate_path_velocity(self):
        """ Recalculate path and velocity fields. May be used when the trajectory is updated. """
        if self._state_list is not None:
            positions = [state.position for state in self._state_list]
            self._path = np.array([[]] if len(self._state_list) == 0 else positions)
        if self._state_list is not None:
            self._velocity = np.array([state.speed for state in self._state_list])

    def add_state(self, new_state: AgentState, reload_path: bool = True):
        """ Add a new state at the end of the trajectory

        Args:
            new_state: AgentState. This should follow the last state of the trajectory in time.
            reload_path: If True then the path and velocity fields are recalculated.
        """
        if len(self._state_list) > 0:
            assert self._state_list[-1].time < new_state.time
        self._state_list.append(new_state)

        if reload_path:
            self._path = np.append(self._path, [new_state.position], axis=1)
            self._velocity = np.append(self._path, new_state.speed)

    def extend(self, trajectory):
        """ Extend the current trajectory with the states of the given trajectory.

        Args:
            trajectory: The given trajectory to use for extension.
        """

        if len(self._state_list) > 0:
            assert self._state_list[-1].time < trajectory.states[0].time
        self._state_list.extend(trajectory.states)
        self.calculate_path_velocity()


class VelocityTrajectory(Trajectory):
    """ Define a trajectory consisting of a 2d path and velocities """

    def __init__(self, path, velocity):
        """ Create a VelocityTrajectory object
        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        super().__init__(path, velocity)
        self._pathlength = self.curvelength(self.path)

    @property
    def pathlength(self) -> np.ndarray:
        """ Length of path travelled at each position along the path in meters. """
        return self._pathlength if self._pathlength is not None else np.array([])

    @property
    def length(self) -> float:
        return self._pathlength[-1]

    @property
    def duration(self) -> float:
        return self.trajectory_times()[-1]

    @property
    def acceleration(self) -> np.ndarray:
        return self.differentiate(self.velocity, self.trajectory_times())

    @property
    def jerk(self) -> np.ndarray:
        return self.differentiate(self.acceleration, self.trajectory_times())
    
    @property
    def angular_velocity(self) -> np.ndarray:
        angle = np.angle(self.path.view(dtype=np.complex128)).reshape(-1)
        return self.differentiate(angle, self.trajectory_times())

    @property
    def angular_acceleration(self) -> np.ndarray:
        return self.differentiate(self.angular_velocity, self.trajectory_times())

    def trajectory_times(self):
        # assume constant acceleration between points on path
        v_avg = (self.velocity[:-1] + self.velocity[1:]) / 2
        s = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(s / v_avg)])
        return t

    def curvelength(self, path):
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)  # Length between points
        return np.cumsum(np.append(0, path_lengths))

    def differentiate(self, x, y):
        """Performs backward difference (since first element is replaced by 0) on data x y"""
        dx = np.diff(x, axis=0)
        dy = np.diff(y, axis=0)
        dx_dy = np.divide(dx , dy)
        dx_dy = np.insert(dx_dy, 0, 0.)
        return dx_dy

class VelocitySmoother:
    """Runs optimisation routine on a VelocityTrajectory object to return realistic velocities according to constraints.
    This accounts for portions of the trajectory where the vehicle is stopped."""

    def __init__(self, trajectory: VelocityTrajectory, n: int = 100, dt_s: float = 0.1,
                 amax_m_s2: float = 5.0, vmin_m_s: float = 1.0, vmax_m_s: float = 10.0, lambda_acc: float = 10.0):
        """ Create a VelocitySmoother object

        Args:
            trajectory: VelocityTrajectory object

        Optional Args:
            n, dt_s, amax_m_s2, vmin_m_s, vmax_m_s, lambda_acc: optimiser parameters
            See @properties for definitions
        """

        self._path = trajectory.path
        self._velocity = trajectory.velocity
        self._pathlength = trajectory.pathlength
        self._n = n
        self._dt = dt_s
        self._amax = amax_m_s2
        self._vmax = vmax_m_s
        self._vmin = vmin_m_s
        self._lambda_acc = lambda_acc
        self._split_velocity = None
        self._split_pathlength = None

    def split_smooth(self, debug: bool = False) -> np.ndarray:
        """Split the trajectory into "go" and "stop" segments, according to vmin and smoothes the "go" segments"""

        self.split_at_stops()

        V_smoothed = []
        for i in range(len(self.split_velocity)):
            if self.split_velocity[i][0] > self.vmin:
                v_smoothed, _, _ = self.smooth_velocity(self.split_pathlength[i], self.split_velocity[i], debug=debug)
                v_smoothed = np.transpose(v_smoothed)
                V_smoothed.extend(v_smoothed[0].tolist())
            else:
                V_smoothed.extend(self.split_velocity[i])
        V_smoothed = np.array(V_smoothed)
        return V_smoothed

    def smooth_velocity(self, pathlength, velocity, debug: bool = False):
        """Creates a linear interpolants for pathlength and velocity, and use them to recursively
        run the optimiser, until the optimisation solution bounds the original pathlength.
        The smoothed velocity is then sampled from the optimisation solution v """

        # TODO: limit n according to dt and expected duration of trajectory segment
        # 1. add a parameter to enable/disable feature
        # 2. calculate trajectory horizon T = sum(pathlength / velocity)
        # 3 limit n with n := min(n, ceil(T/dt)) 

        # TODO: optimiser runs again without velocity constraints if not convergence
        # - remove max velocity constraint
        # - remove velocity match at start of trajectory
        # optimiser runs a total of X times before timeout

        # Create interpolants for pathlength and velocity
        ind = list(range(len(pathlength)))
        pathlength_interpolant = self.lin_interpolant(ind, pathlength)
        velocity_interpolant = self.lin_interpolant(pathlength, velocity)

        X = [pathlength[0]]
        V = [velocity[0]]
        while X[-1] < pathlength[-1]:
            x, v = self.optimiser(pathlength, velocity, X[-1], V[-1],
                                  pathlength_interpolant, velocity_interpolant,
                                  debug=debug)
            X.extend(list(x[1:]))
            V.extend(list(v[1:]))

        X = np.array(X)
        V = np.array(V)
        X, V = self.remove_duplicates(X, V)
        sol_interpolant = self.lin_interpolant(X, V)

        return np.array(sol_interpolant(pathlength)), X, V

    def optimiser(self, pathlength, velocity, x_start: float, v_start: float,
                  pathlength_interpolant, velocity_interpolant, debug: bool = False):

        opti = ca.Opti()

        # Create optimisation variables
        x = opti.variable(self.n)
        v = opti.variable(self.n)

        # Initialise optimisation variables
        # TODO: better process to initialise variables
        # 1. estimate the horizon T for a specific n, dt from original data
        # 2. initialise pathlength, velocities using indices corresponding to horizon T in original data

        if x_start == pathlength[0]:
            ind_start = 0
        else:
            # TODO refactor into a function (use binary search for efficiency improvement?)
            for i, el in enumerate(pathlength):
                if el > x_start:
                    ind_larger = i
                    break
            ind_start = ind_larger - 1 + (x_start - pathlength[ind_larger - 1]) / (
                        pathlength[ind_larger] - pathlength[ind_larger - 1])

        ind_n = np.linspace(ind_start, len(pathlength), self.n)

        path_ini = pathlength_interpolant(ind_n)
        vel_ini = velocity_interpolant(path_ini)
        # vel_ini = [min(velocity)] * self.n #may help convergence in some cases
        opti.set_initial(x, path_ini)
        opti.set_initial(v, vel_ini)

        # Optimisation objective to minimise
        J = ca.sumsqr(v - velocity_interpolant(x)) + self.lambda_acc * ca.sumsqr(v[1:] - v[:-1])
        opti.minimize(J)

        # Optimisation constraints
        opti.subject_to(x[0] == x_start)
        opti.subject_to(v[0] == v_start)
        opti.subject_to(opti.bounded(self.vmin, v, self.vmax))
        opti.subject_to(v <= velocity_interpolant(x))

        for k in range(0, self.n - 1):
            opti.subject_to(x[k + 1] == x[k] + v[k] * self.dt)
            opti.subject_to(ca.fabs(v[k + 1] - v[k]) < self.amax * self.dt)

        # Solve
        opts = {}
        if not (debug):  # disable terminal printout
            opts['ipopt.print_level'] = 0
            opts['print_time'] = 0
            opts['ipopt.sb'] = "yes"

        opti.solver('ipopt', opts)
        sol = opti.solve()

        return sol.value(x), sol.value(v)

    def lin_interpolant(self, x, y):
        """Creates a differentiable Casadi interpolant object to linearly interpolate y from x"""
        return ca.interpolant('LUT', 'linear', [x], y)

    def split_at_stops(self):
        """Split original velocity and pathlength arrays, 
        for the optimisation process to run separately on each splitted array. 
        The splitted array corresponds of trajectory segments between stops.
        The function will also remove any extended stops from the splitted arrays 
        (parts where consecutive velocities points are 0"""

        split = [i + 1 for i in range(len(self.velocity) - 1)
                 if (self.velocity[i] <= self.vmin < self.velocity[i + 1])
                 or (self.velocity[i] > self.vmin >= self.velocity[i + 1])]
        if self.velocity[-1] <= self.vmin: split.append(len(self.velocity) - 1)

        self._split_velocity = [x for x in np.split(self.velocity, split) if len(x != 0)]
        self._split_pathlength = [x for x in np.split(self.pathlength, split) if len(x != 0)]

    def remove_duplicates(self, x, v):
        dup_ind = np.array(np.where(v == 0.0)[0])
        dup_ind += 1
        x = np.delete(x, dup_ind)
        v = np.delete(v, dup_ind)
        return x, v

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
    def vmin(self) -> float:
        """Returns the minimum velocity limit used in smoothing optimisation in m/s. 
        Vehicle will be considered at a stop for velocities below this value.
        The optimisation smoothing will not be run at points below this velocity"""
        return self._vmin

    @property
    def vmax(self) -> float:
        """Returns the maximum velocity limit used in smoothing optimisation in m/s"""
        return self._vmax

    @property
    def lambda_acc(self) -> float:
        """Returns the lambda parameter used to control the weighting of the 
        acceleration penalty in smoothing optimisation"""
        return self._lambda_acc

    @property
    def path(self) -> np.ndarray:
        """Returns the xy position at each trajectory waypoint in meters"""
        return self._path

    @property
    def velocity(self) -> np.ndarray:
        """Returns the velocity at each trajectory waypoint in m/s"""
        return self._velocity

    @property
    def pathlength(self) -> np.ndarray:
        """Returns the cummulative length travelled in the trajectory in meters"""
        return self._pathlength

    @property
    def split_velocity(self) -> np.ndarray:
        """Returns the velocity at each trajectory waypoint in m/s"""
        return self._split_velocity

    @property
    def split_pathlength(self) -> np.ndarray:
        """Returns the cummulative length travelled in the trajectory in meters"""
        return self._split_pathlength
