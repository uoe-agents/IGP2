import casadi as ca
import math
import numpy as np
import logging
from typing import Tuple

from igp2.core.trajectory import VelocityTrajectory

logger = logging.getLogger(__name__)


class VelocitySmoother:
    """Runs optimisation routine on a VelocityTrajectory object to return realistic velocities according to constraints.
    This accounts for portions of the trajectory where the vehicle is stopped."""

    def __init__(self,
                 n: int = 100,
                 dt_s: float = 0.1,
                 amax_m_s2: float = 5.0,
                 vmin_m_s: float = 1.0,
                 vmax_m_s: float = 10.0,
                 lambda_acc: float = 10.0,
                 horizon_threshold: float = 1.2,
                 min_n: int = 5):
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
        """ Split the trajectory into "go" and "stop" segments, according to vmin and smoothes the "go" segments """

        self.split_at_stops()
        if len(self._split_velocity) > 1:
            logger.debug(f"Stops detected. Splitting trajectory into {len(self.split_velocity)} segments.")

        V_smoothed = []
        for i in range(len(self.split_velocity)):
            if len(self._split_velocity) > 1:
                logger.debug(f"Smoothing segment {i}.")
            if len(self._split_velocity[i]) < 2:
                logger.debug(f"Cannot smooth velocity profile of length 1. ")
            if len(self._split_velocity[i]) >= 2 and self.split_velocity[i][0] > self.vmin:
                v_smoothed, _, _ = self.smooth_velocity(self.split_pathlength[i], self.split_velocity[i], debug=debug)
                v_smoothed = np.transpose(v_smoothed)
                V_smoothed.extend(v_smoothed[0].tolist())
            else:
                V_smoothed.extend(self.split_velocity[i])
        V_smoothed = np.array(V_smoothed)
        return V_smoothed

    def smooth_velocity(self, pathlength: np.ndarray, velocity: np.ndarray, debug: bool = False):
        """Creates a linear interpolants for pathlength and velocity, and use them to recursively
        run the optimiser, until the optimisation solution bounds the original pathlength.
        The smoothed velocity is then sampled from the optimisation solution v """

        options = [{"uniform_initialisation": False, "disable_upper_bound": False, "disable_vmax": False,
                    "disable_amax": False, "disable_lambda": False, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": True, "disable_upper_bound": False, "disable_vmax": False,
                    "disable_amax": False, "disable_lambda": False, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": False, "disable_upper_bound": True, "disable_vmax": False,
                    "disable_amax": False, "disable_lambda": False, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": True, "disable_upper_bound": True, "disable_vmax": False,
                    "disable_amax": False, "disable_lambda": False, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": False, "disable_upper_bound": True, "disable_vmax": True,
                    "disable_amax": True, "disable_lambda": False, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": True, "disable_upper_bound": True, "disable_vmax": True,
                    "disable_amax": True, "disable_lambda": False, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": False, "disable_upper_bound": True, "disable_vmax": True,
                    "disable_amax": True, "disable_lambda": True, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": True, "disable_upper_bound": True, "disable_vmax": True,
                    "disable_amax": True, "disable_lambda": True, "disable_v0": False, "debug": debug},

                   {"uniform_initialisation": False, "disable_upper_bound": True, "disable_vmax": True,
                    "disable_amax": True, "disable_lambda": True, "disable_v0": True, "debug": debug},

                   {"uniform_initialisation": True, "disable_upper_bound": True, "disable_vmax": True,
                    "disable_amax": True, "disable_lambda": True, "disable_v0": True, "debug": debug}]

        def recursive_optimisation(inc: int = 0):
            try:
                opt = options[inc]
            except IndexError as e:
                logger.debug(
                    "Optimiser did not converge with any of the relaxation attempts, Appending unsmoothed velocity.")
                x = pathlength[math.floor(ind_start):]
                v = velocity[math.floor(ind_start):]
                return x, v
            try:
                logger.debug(f"Trying velocity smoothing with {str(options[inc])}")
                return self.optimiser(n, velocity_interpolant, pathvel_interpolant, ind_start, ind_end, X[-1], V[-1],
                                      options=opt)
            except RuntimeError as e:
                logger.debug(f"Optimiser did not converge with {str(e)}")
                inc += 1
                return recursive_optimisation(inc)

        # Create interpolants for pathlength and velocity
        ind = list(range(len(pathlength)))
        ind_end = ind[-1]
        velocity_interpolant = self.lin_interpolant(ind, velocity)
        pathvel_interpolant = self.lin_interpolant(pathlength, velocity)

        X = [pathlength[0]]
        V = [velocity[0]]
        count = 0
        while X[-1] < pathlength[-1]:
            if count > 0:
                logger.debug("Solution is not bounding trajectory, extending optimisation problem.")
            count += 1
            ind_start = self._find_ind_start(X[-1], pathlength)
            t = np.sum(self._trajectory.timesteps[math.floor(ind_start):])
            n = max(self.min_n, min(self.n, math.ceil(t / self.dt * self.horizon_threshold)))
            if n != self.n: logger.debug(f"Higher than necessary n detected. using n = {n} instead")
            x, v = recursive_optimisation()
            X.extend(list(x[1:]))
            V.extend(list(v[1:]))

        X = np.array(X)
        V = np.array(V)
        X, V = self.remove_duplicates(X, V)
        sol_interpolant = self.lin_interpolant(X, V)

        return np.array(sol_interpolant(pathlength)), X, V

    def optimiser(self, n: int, velocity_interpolant: ca.interpolant, pathvel_interpolant: ca.interpolant,
                  ind_start: float, ind_end: float, x_start: float, v_start: float, options: dict()):

        if options["disable_vmax"]:
            vmax = 1e2
        else:
            vmax = self.vmax
        if options["disable_amax"]:
            amax = 1e3
        else:
            amax = self.amax
        if options["disable_lambda"]:
            lambda_acc = 0
        else:
            lambda_acc = self.lambda_acc

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
                path_ini[i] = path_ini[i - 1] + vel_ini[i] * self.dt
            else:
                path_ini[i] = path_ini[i - 1] + 0.5 * (vel_ini[i - 1] + vel_ini[i + 1]) * self.dt
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

        opts['ipopt.max_iter'] = 30  # reduce max number of iterations to improve inference time.

        opti.solver('ipopt', opts)
        sol = opti.solve()

        return sol.value(x), sol.value(v)

    def _find_ind_start(self, x_start: float, x: np.ndarray) -> float:
        """Finds the position corresponding to x_start value in x array. 
        Linearly interpolates between two positions if not exact match."""

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

    def lin_interpolant(self, x: np.ndarray, y: np.ndarray) -> ca.interpolant:
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

    def remove_duplicates(self, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Removes any duplicates in pathlength and velocity caused by consecutive 0 velocities"""
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
