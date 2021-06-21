import abc
import numpy as np
import casadi as ca
import math
import logging
from typing import Union, Tuple, List, Dict, Optional

from typing import List

from numpy.lib.function_base import diff

from igp2.agent import AgentState
from igp2.util import get_curvature

logger = logging.getLogger(__name__)

class Trajectory(abc.ABC):
    """ Base class for all Trajectory objects """

    def __init__(self, path: np.ndarray = None, velocity: np.ndarray = None, velocity_stop: float = 0.1):
        """ Create an empty Trajectory object

        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        self._path = path
        self._velocity = velocity
        self._velocity_stop = velocity_stop

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
        var = self.differentiate(self.velocity, self.trajectory_times())
        var = np.where(abs(var) >= 1e2, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def jerk(self) -> np.ndarray:
        var = self.differentiate(self.acceleration, self.trajectory_times())
        var = np.where(abs(var) >= 1e3, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Calculates angular velocity, handling discontinuity at theta = pi"""
        dheading = np.pi - np.abs(np.pi - np.abs(np.diff(self.heading)) % (2 * np.pi))
        var = self.differentiate(None, self.trajectory_times(), dx=dheading)
        var = np.where(abs(var) >= 1e1, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def angular_acceleration(self) -> np.ndarray:
        var = self.differentiate(self.angular_velocity, self.trajectory_times())
        var = np.where(abs(var) >= 1e3, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def curvature(self) -> np.ndarray:
        var = np.nan_to_num(get_curvature(self.path), posinf=0.0, neginf=0.0)
        var = np.where(abs(var) >= 1e3, 0., var)  # takeout extreme values due to numerical errors / bad data
        return np.where(self.velocity <= self.velocity_stop, 0., var)

    @property
    def velocity_stop(self) -> float:
        """Velocity at or under which the vehicle is considered to be at a stop"""
        return self._velocity_stop

    @property
    def initial_agent_state(self) -> AgentState:
        return AgentState(0, self.path[0], self.velocity[0], self.acceleration[0], self.heading[0])

    @property
    def final_agent_state(self) -> AgentState:
        return AgentState(0, self.path[-1], self.velocity[-1], self.acceleration[-1], self.heading[-1])

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

    @property
    def heading(self) -> np.ndarray:
        """ Heading of vehicle in radians.
        
        Returns:
             array of heading value from dataset, or estimate from trajectory 
             path if heading data is not available
        """
        raise NotImplementedError

    def differentiate(self, x: np.ndarray, y: np.ndarray, dx: np.ndarray = None, dy: np.ndarray = None):
        """Performs backward difference on data x y. 
        First element is computed by forward difference. (continuity assumption)
        Can overload dx and dy if required.
        Will replace nonsensical values (Nan and +/- inf with 0.0)"""
        if dx is None: dx = np.diff(x, axis=0)
        if dy is None: dy = np.diff(y, axis=0)
        dx_dy = np.divide(dx, dy)
        dx_dy = np.insert(dx_dy, 0, dx_dy[0])
        return np.nan_to_num(dx_dy, posinf=0.0, neginf=0.0)

    def heading_from_path(self, path) -> np.ndarray:
        dpath = np.diff(path, axis=0)
        heading = np.angle(dpath.view(dtype=np.complex128)).reshape(-1)
        heading = np.insert(heading, 0, heading[0])
        return heading

    def trajectory_dt(self):
        # assume constant acceleration between points on path
        v_avg = (self.velocity[:-1] + self.velocity[1:]) / 2
        s = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        dt = np.concatenate([[0], s / v_avg])
        return dt

    def trajectory_times(self):
        return np.cumsum(self.trajectory_dt())

    def extend(self, new_trajectory):
        raise NotImplementedError


class StateTrajectory(Trajectory):
    """ Implements a Trajectory that is build discreet observations at each time step. """

    def __init__(self, fps: int, start_time: int, frames: List[AgentState] = None,
                 path: np.ndarray = None, velocity: np.ndarray = None, velocity_stop: float = 0.1):
        """ Create a new StateTrajectory

        Args:
            fps: The number of time steps each second
            start_time: The first time step of the StateTrajectory
            frame: Optionally, specify a list of AgentStates with an observed trajectory
            path: Path points along the trajectory. Ignored.
            velocity: Velocities at each path point. Ignored.
        """
        super().__init__(path, velocity, velocity_stop=velocity_stop)
        self.fps = fps
        self.start_time = start_time
        self._state_list = frames if frames is not None else []
        self.calculate_path_and_velocity()

    def __getitem__(self, item: int) -> AgentState:
        return self._state_list[item]

    def __iter__(self):
        yield from self._state_list

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
        """ Add a new state at the end of the trajectory

        Args:
            new_state: AgentState. This should follow the last state of the trajectory in time.
            reload_path: If True then the path and velocity fields are recalculated.
        """
        if len(self._state_list) > 0:
            assert self._state_list[-1].time < new_state.time
        self._state_list.append(new_state)

        if reload_path:
            if self._path is None or self._velocity is None:
                self.calculate_path_and_velocity()
            else:
                self._path = np.append(self._path, np.array([new_state.position]), axis=0)
                self._velocity = np.append(self._velocity, new_state.speed)

    def extend(self, trajectory):
        """ Extend the current trajectory with the states of the given trajectory.

        Args:
            trajectory: The given trajectory to use for extension.
        """

        if len(self._state_list) > 0:
            assert self._state_list[-1].time < trajectory.states[0].time
        self._state_list.extend(trajectory.states)
        self.calculate_path_and_velocity()

    def slice(self, start_idx: int, end_idx: int):
        return StateTrajectory(self.fps, self.start_time,
                               self._state_list[start_idx:end_idx],
                               self.path[start_idx:end_idx],
                               self.velocity[start_idx:end_idx],
                               self.velocity_stop)


class VelocityTrajectory(Trajectory):
    """ Define a trajectory consisting of a 2d path and velocities """

    def __init__(self, path: np.ndarray, velocity: np.ndarray, heading: np.ndarray = None, velocity_stop: float = 0.1):
        """ Create a VelocityTrajectory object
        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        super().__init__(path, velocity, velocity_stop=velocity_stop)
        self._pathlength = self.curvelength(path)
        if heading is None: self._heading = self.heading_from_path(self.path)
        else: self._heading = heading

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
    def heading(self) -> np.ndarray:
        return self._heading

    def curvelength(self, path) -> np.ndarray:
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)  # Length between points
        return np.cumsum(np.append(0, path_lengths))

    def insert(self, trajectory: Trajectory):
        """Inserts a Trajectory at the begining of the VelocityTrajectory object, removing the first element of the original VelocityTrajectory."""
        if trajectory.velocity.size == 0:
            pass
        else:
            path = np.concatenate((trajectory.path, self.path[1:]))
            velocity = np.concatenate((trajectory.velocity, self.velocity[1:]))
            heading = np.concatenate((trajectory.heading, self.heading[1:]))
            self.__init__(path, velocity, heading, self._velocity_stop)

    def extend(self, new_trajectory):
        if isinstance(new_trajectory, Trajectory):
            path_heading = np.concatenate([self.path[-1], new_trajectory.path], axis=0)
            self._path = np.concatenate([self.path, new_trajectory.path], axis=0)
            self._velocity = np.concatenate([self.velocity, new_trajectory.velocity])
        else:
            path_heading = np.concatenate([[self.path[-1]], new_trajectory[0]], axis=0)
            self._path = np.concatenate([self.path, new_trajectory[0]], axis=0)
            self._velocity = np.concatenate([self.velocity, new_trajectory[1]])
        heading = self.heading_from_path(path_heading)
        self._heading = np.concatenate([self.heading, heading[1:]])
        self._pathlength = self.curvelength(self._path)

class VelocitySmoother:
    """Runs optimisation routine on a VelocityTrajectory object to return realistic velocities according to constraints.
    This accounts for portions of the trajectory where the vehicle is stopped."""

    def __init__(self, n: int = 100, dt_s: float = 0.1,
                 amax_m_s2: float = 5.0, vmin_m_s: float = 1.0, vmax_m_s: float = 10.0, lambda_acc: float = 10.0, horizon_threshold: float = 1.2, min_n: int = 5):
        """ Create a VelocitySmoother object

        Args:
            trajectory: VelocityTrajectory object

        Optional Args:
            n, dt_s, amax_m_s2, vmin_m_s, vmax_m_s, lambda_acc: optimiser parameters
            See @properties for definitions
        """

        self._n = n
        self._min_n = min_n
        self._dt = dt_s
        self._amax = amax_m_s2
        self._vmax = vmax_m_s
        self._vmin = vmin_m_s
        self._lambda_acc = lambda_acc
        self._horizon_threshold = horizon_threshold
        self._split_velocity = None
        self._split_pathlength = None

    def load_trajectory(self, trajectory: VelocityTrajectory):

        self._trajectory = trajectory

    def split_smooth(self, debug: bool = False) -> np.ndarray:
        """Split the trajectory into "go" and "stop" segments, according to vmin and smoothes the "go" segments"""

        self.split_at_stops()
        if len(self._split_velocity) > 1:
            logger.debug(f"Stops detected. Splitting trajectory into {len(self.split_velocity)} segments.")

        V_smoothed = []
        for i in range(len(self.split_velocity)):
            if len(self._split_velocity) > 1:
                logger.debug(f"Smoothing segment {i}.")
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

        options = {"debug": debug, "uniform_initialisation": False, "disable_upper_bound": False, "disable_vmax": False,
        "disable_amax": False, "disable_lambda": False, "disable_v0": False}

        # Create interpolants for pathlength and velocity
        ind = list(range(len(pathlength)))
        ind_end = ind[-1]
        velocity_interpolant = self.lin_interpolant(ind, velocity)
        pathvel_interpolant = self.lin_interpolant(pathlength, velocity)

        X = [pathlength[0]]
        V = [velocity[0]]
        count = 0
        while X[-1] < pathlength[-1]:
            if count > 0 : 
                logger.debug("Solution is not bounding trajectory, extending optimisation problem.")
            count += 1
            ind_start = self._find_ind_start(X[-1], pathlength)
            t = np.sum(self._trajectory.trajectory_dt()[math.floor(ind_start):])
            n = max(self.min_n, min(self.n, math.ceil(t/self.dt * self.horizon_threshold)))
            if n!=self.n: logger.debug(f"Higher than necessary n detected. using n = {n} instead")
            try:
                x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                    options=options)
            except RuntimeError as e:
                logger.debug(f"Optimiser did not converge with {str(e)}")
                logger.debug("Retrying with low velocity initialisation.")
                options["uniform_initialisation"] = True
                try:
                    x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                    options=options)
                except RuntimeError as e:
                    logger.debug(f"Optimiser did not converge with {str(e)}")
                    logger.debug("Retrying with no upper bound.")
                    options["uniform_initialisation"] = False
                    options["disable_upper_bound"] = True
                    try:
                        x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                        options=options)
                    except RuntimeError as e:
                        logger.debug(f"Optimiser did not converge with {str(e)}")
                        logger.debug("Retrying with no upper bound and low velocity initialisation.")
                        options["uniform_initialisation"] = True
                        try:
                            x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                            options=options)
                        except RuntimeError as e:
                            logger.debug(f"Optimiser did not converge with {str(e)}")
                            logger.debug("Retrying with no vmax and amax constraints.")
                            options["uniform_initialisation"] = False
                            options["disable_vmax"] = True
                            options["disable_amax"] = True
                            try:
                                x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                            options=options)
                            except RuntimeError as e:
                                logger.debug(f"Optimiser did not converge with {str(e)}")
                                logger.debug("Retrying with no vmax and amax constraints and low velocity initialisation.")
                                options["uniform_initialisation"] = True
                                try:
                                    x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                            options=options)
                                except RuntimeError as e:
                                    logger.debug(f"Optimiser did not converge with {str(e)}")
                                    logger.debug("Retrying with lambda = 0")
                                    options["disable_lambda"] = True
                                    options["uniform_initialisation"] = False
                                    try:
                                        x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                                options=options)
                                    except RuntimeError as e:
                                        logger.debug(f"Optimiser did not converge with {str(e)}")
                                        logger.debug("Retrying with lambda = 0 and low velocity initialisation.")
                                        options["uniform_initialisation"] = True
                                        try:
                                            x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                                options=options)
                                        except RuntimeError as e:
                                            logger.debug(f"Optimiser did not converge with {str(e)}")
                                            logger.debug("Retrying with no v[0] constraint")
                                            options["disable_v0"] = True
                                            options["uniform_initialisation"] = False
                                            try:
                                                x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                                options=options)
                                            except RuntimeError as e:
                                                logger.debug(f"Optimiser did not converge with {str(e)}")
                                                logger.debug("Retrying with no v[0] constraint and low velocity initialisation.")
                                                options["uniform_initialisation"] = True
                                                try:
                                                    x, v = self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                                options=options)
                                                except RuntimeError as e:
                                                    logger.debug(f"Optimiser did not converge with {str(e)}")
                                                    logger.debug("Appending unsmoothed velocity")
                                                    x = pathlength[math.floor(ind_start):]
                                                    v = velocity[math.floor(ind_start):]
            X.extend(list(x[1:]))
            V.extend(list(v[1:]))

        X = np.array(X)
        V = np.array(V)
        X, V = self.remove_duplicates(X, V)
        sol_interpolant = self.lin_interpolant(X, V)

        return np.array(sol_interpolant(pathlength)), X, V

    def optimiser(self, n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, x_start, v_start, options: dict()):

        if options["disable_vmax"]: vmax = 1e2
        else: vmax = self.vmax
        if options["disable_amax"]: amax = 1e3
        else: amax = self.amax
        if options["disable_lambda"]: lambda_acc = 0
        else: lambda_acc = self.lambda_acc
        
        opti = ca.Opti()

        # Create optimisation variables
        x = opti.variable(n)
        v = opti.variable(n)

        ind_n = np.linspace(ind_start, ind_end, n)
        if options["uniform_initialisation"]:
            vel_ini = [v_start] * n
        else:
            vel_ini = velocity_interpolant(ind_n)

        if not options["disable_v0"]:
            vel_ini[0] = v_start

        path_ini = np.empty(n, float)
        path_ini[0] = x_start
        for i in range(1, len(path_ini)):
            if i == len(path_ini) - 1:
                path_ini[i] = path_ini[i-1] + vel_ini[i] * self.dt
            else:
                path_ini[i] = path_ini[i-1] + 0.5 * (vel_ini[i-1] + vel_ini[i+1]) * self.dt
        opti.set_initial(x, path_ini)
        opti.set_initial(v, vel_ini)

        # Optimisation objective to minimise
        J = ca.sumsqr(v - pathvel_interpolant(x)) + lambda_acc * ca.sumsqr(v[1:] - v[:-1])
        opti.minimize(J)

        # Optimisation constraints
        opti.subject_to(x[0] == x_start)
        if not options["disable_v0"]:
            opti.subject_to(v[0] == v_start)
        opti.subject_to(opti.bounded(self.vmin, v, vmax))
        if not options["disable_upper_bound"]:
            opti.subject_to(v <= pathvel_interpolant(x))

        for k in range(0, n - 1):
            opti.subject_to(x[k + 1] == x[k] + v[k] * self.dt)
            opti.subject_to(ca.fabs(v[k + 1] - v[k]) < amax * self.dt)

        # Solve
        opts = {}
        if not (options["debug"]):  # disable terminal printout
            opts['ipopt.print_level'] = 0
            opts['print_time'] = 0
            opts['ipopt.sb'] = "yes"

        opts['ipopt.max_iter'] = 30 #reduce max number of iterations to improve inference time.

        opti.solver('ipopt', opts)
        sol = opti.solve()

        return sol.value(x), sol.value(v)

    def _find_ind_start(self, x_start: float, x: np.ndarray):

        if x_start == x[0]:
            ind_start = 0.
        else:
            for i, el in enumerate(x):
                if el > x_start:
                    ind_larger = i
                    break
            ind_start = ind_larger - 1 + (x_start - x[ind_larger - 1]) / (
                    x[ind_larger] - x[ind_larger - 1])

        return ind_start

    def lin_interpolant(self, x, y):
        """Creates a differentiable Casadi interpolant object to linearly interpolate y from x"""
        return ca.interpolant('LUT', 'linear', [x], y)

    def split_at_stops(self):
        """Split original velocity and pathlength arrays, 
        for the optimisation process to run separately on each splitted array. 
        The splitted array corresponds of trajectory segments between stops.
        The function will also remove any extended stops from the splitted arrays 
        (parts where consecutive velocities points are 0"""

        split = [i + 1 for i in range(len(self._trajectory.velocity) - 1)
                 if (self._trajectory.velocity[i] <= self.vmin < self._trajectory.velocity[i + 1])
                 or (self._trajectory.velocity[i] > self.vmin >= self._trajectory.velocity[i + 1])]
        if self._trajectory.velocity[-1] <= self.vmin: split.append(len(self._trajectory.velocity) - 1)

        self._split_velocity = [x for x in np.split(self._trajectory.velocity, split) if len(x != 0)]
        self._split_pathlength = [x for x in np.split(self._trajectory.pathlength, split) if len(x != 0)]

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
    def horizon_threshold(self) -> float:
        """Returns the multiple of the estimated trajectory horizon that is used to 
        limit the maximum value of n in the optimiser"""
        return self._horizon_threshold

    @property
    def min_n(self) -> int:
        """Returns minimum value for n"""
        return self._min_n

    @property
    def split_velocity(self) -> np.ndarray:
        """Returns the velocity at each trajectory waypoint in m/s"""
        return self._split_velocity

    @property
    def split_pathlength(self) -> np.ndarray:
        """Returns the cummulative length travelled in the trajectory in meters"""
        return self._split_pathlength
