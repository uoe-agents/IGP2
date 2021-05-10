import numpy as np
import casadi as ca
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
        self.pathlength = self.curvelength(self.path)

    def curvelength(self, path):
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1) # Length between points
        return np.cumsum(np.append(0, path_lengths))

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

    def split_smooth(self, debug: bool = False):
        """Split the trajectory into "go" and "stop" segments, according to vmin and smoothes the "go" segments"""

        self.split_at_stops()

        V_smoothed = []
        for i in range(len(self.split_velocity)):
            if self.split_velocity[i][0] > self.vmin:
                v_smoothed , _ , _ = self.smooth_velocity(self.split_pathlength[i], self.split_velocity[i], debug = debug)
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

        #TODO: limit n according to dt and expected duration of trajectory segment
        # 1. add a parameter to enable/disable feature
        # 2. calculate trajectory horizon T = sum(pathlength / velocity)
        # 3 limit n with n := min(n, ceil(T/dt)) 

        # Create interpolants for pathlength and velocity
        ind = list(range(len(pathlength)))
        pathlength_interpolant = self.lin_interpolant(ind, pathlength)
        velocity_interpolant = self.lin_interpolant(pathlength, velocity)

        X = [pathlength[0]]
        V = [velocity[0]]
        while X[-1] < pathlength[-1]:
            x , v = self.optimiser(pathlength, velocity, X[-1], V[-1], pathlength_interpolant, velocity_interpolant, debug = debug)
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

        if x_start == pathlength[0] : ind_start = 0
        else:
            #TODO refactor into a function (use binary search for efficiency improvement?)
            for i, el in enumerate(pathlength):
                if el > x_start:
                    ind_larger = i
                    break
            ind_start = ind_larger - 1 + (x_start - pathlength[ind_larger - 1])/ (pathlength[ind_larger] - pathlength[ind_larger - 1])

        ind_n = np.linspace(ind_start, len(pathlength), self.n)
        path_ini = pathlength_interpolant(ind_n)
        vel_ini = velocity_interpolant(path_ini)
        #vel_ini = [min(velocity)] * self.n #may help convergence in some cases
        opti.set_initial(x, path_ini)
        opti.set_initial(v, vel_ini)

        # Optimisation objective to minimise
        J = ca.sumsqr(v - velocity_interpolant(x)) + self.lambda_acc * ca.sumsqr(v[1:]-v[:-1])
        opti.minimize( J )

        # Optimisation constraints
        opti.subject_to( x[0] == x_start)
        opti.subject_to( v[0] == v_start)
        opti.subject_to( opti.bounded(self.vmin, v, self.vmax))
        opti.subject_to(v <= velocity_interpolant(x))

        for k in range(0 , self.n - 1):
            opti.subject_to( x[k+1] == x[k] + v[k] * self.dt )
            opti.subject_to( ca.fabs(v[k + 1] - v[k]) < self.amax * self.dt )

        # Solve
        opts = {}
        if not(debug): #disable terminal printout
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
            if (self.velocity[i] <= self.vmin and self.velocity[i+1] > self.vmin) 
            or (self.velocity[i] > self.vmin and self.velocity[i+1] <= self.vmin)]
        if self.velocity[-1] <= self.vmin: split.append(len(self.velocity) - 1)
        
        self._split_velocity = [x for x in np.split(self.velocity, split) if len(x!=0)]
        self._split_pathlength = [x for x in np.split(self.pathlength, split) if len(x!=0)]

    def remove_duplicates(self, x, v):
        dup_ind = np.array(np.where(v==0.0)[0])
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
    def path(self):
        """Returns the xy position at each trajectory waypoint in meters"""
        return self._path

    @property
    def velocity(self):
        """Returns the velocity at each trajectory waypoint in m/s"""
        return self._velocity

    @property
    def pathlength(self):
        """Returns the cummulative length travelled in the trajectory in meters"""
        return self._pathlength

    @property
    def split_velocity(self):
        """Returns the velocity at each trajectory waypoint in m/s"""
        return self._split_velocity

    @property
    def split_pathlength(self):
        """Returns the cummulative length travelled in the trajectory in meters"""
        return self._split_pathlength