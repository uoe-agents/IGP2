import numpy as np
import more_itertools as mit
from typing import Dict, Optional
from numpy.core.function_base import linspace
from scipy.interpolate import splev, splprep

from igp2.core.trajectory import Trajectory, VelocityTrajectory, StateTrajectory
from igp2.core.goal import Goal

class Cost:
    """ Define the exact cost signal of a trajectory.
    The IGP2 paper refers to this as reward, which can be interpreted as negative cost. """

    def __init__(self, factors: Dict[str, float] = None, limits: Dict[str, float] = None):
        """ Initialise a new Cost class with the given weights.

        Args:
            factors: dictionary of weights for different costs, comprising of
                w_time: Time to goal weight
                w_vel: Velocity weight
                w_acc: Acceleration weight
                w_jerk: Jerk weight
                w_head: Heading weight
                w_angvel: Angular velocity weight
                w_angacc: Angular acceleration weight
                w_curv: Curvature weight
                w_saf: Safety weight (not implemented)
            limits: dictionary of limits for different costs, comprising of

        """
        self._factors = {"time": 1., "velocity": 1., "acceleration": 1., "jerk": 1., "heading": 1.,
                         "angular_velocity": 1.,
                         "angular_acceleration": 1., "curvature": 1., "safety": 1.} if factors is None else factors

        self._limits = {"velocity": 19.214, "acceleration": 4.3442, "jerk": 125.63, "heading": 2 * np.pi,
                        "angular_velocity": 1.013,
                        "angular_acceleration": 35.127, "curvature": 108.04} if limits is None else limits

        self.COMPONENTS = list(self._factors)

        self._cost = None
        self._dcost = None
        self._components = None

    def trajectory_cost(self,
                        trajectory: Trajectory,
                        goal: Goal) -> float:
        """ Calculate the total cost of the trajectory given a goal.

        Args:
            trajectory: The trajectory to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost value
        """
        if isinstance(trajectory, StateTrajectory):
            trajectory = VelocityTrajectory(trajectory.path, trajectory.velocity,
                                               trajectory.heading, trajectory.timesteps)

        goal_reached_i = self._goal_reached(trajectory, goal)

        self._components = {
            "time": abs(self._time_to_goal(trajectory, goal_reached_i)),
            "velocity": abs(self._velocity(trajectory, goal_reached_i)),
            "acceleration": abs(self._longitudinal_acceleration(trajectory, goal_reached_i)),
            "jerk": abs(self._longitudinal_jerk(trajectory, goal_reached_i)),
            "heading": abs(self._heading(trajectory, goal_reached_i)),
            "angular_velocity": abs(self._angular_velocity(trajectory, goal_reached_i)),
            "angular_acceleration": abs(self._angular_acceleration(trajectory, goal_reached_i)),
            "curvature": abs(self._curvature(trajectory, goal_reached_i))
        }

        self._cost = sum([self._factors[component] * cost for component, cost in self._components.items()])

        return self._cost

    def cost_difference(self, trajectory1: Trajectory, trajectory2: Trajectory, goal: Goal) -> float:
        """ Calculate the sum of the cost elements differences between two trajectories, given a goal.
        This is not a function that is used in the current implementation.

        Args:
            trajectory1: The first trajectory to examine
            trajectory2: The second trajectory to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost difference value
        """
        goal_reached_i1 = self._goal_reached(trajectory1, goal)
        goal_reached_i2 = self._goal_reached(trajectory2, goal)

        dcost_time_to_goal = abs(
            self._time_to_goal(trajectory1, goal_reached_i1) - self._time_to_goal(trajectory2, goal_reached_i2))
        dcost_velocity = abs(
            self._velocity(trajectory1, goal_reached_i1) - self._velocity(trajectory2, goal_reached_i2))
        dcost_longitudinal_acceleration = abs(
            self._longitudinal_acceleration(trajectory1, goal_reached_i1) - self._longitudinal_acceleration(trajectory2,
                                                                                                            goal_reached_i2))
        dcost_longitudinal_jerk = abs(
            self._longitudinal_jerk(trajectory1, goal_reached_i1) - self._longitudinal_jerk(trajectory2,
                                                                                            goal_reached_i2))
        dcost_heading = abs(self._heading(trajectory1, goal_reached_i1) - self._heading(trajectory2, goal_reached_i2))
        dcost_angular_velocity = abs(
            self._angular_velocity(trajectory1, goal_reached_i1) - self._angular_velocity(trajectory2, goal_reached_i2))
        dcost_angular_acceleration = abs(
            self._angular_acceleration(trajectory1, goal_reached_i1) - self._angular_acceleration(trajectory2,
                                                                                                  goal_reached_i2))
        dcost_curvature = abs(
            self._curvature(trajectory1, goal_reached_i1) - self._curvature(trajectory2, goal_reached_i2))

        self._cost = (self.factors["time"] * dcost_time_to_goal +
                      self.factors["velocity"] * dcost_velocity +
                      self.factors["acceleration"] * dcost_longitudinal_acceleration +
                      self.factors["jerk"] * dcost_longitudinal_jerk +
                      self.factors["heading"] * dcost_heading +
                      self.factors["angular_velocity"] * dcost_angular_velocity +
                      self.factors["angular_acceleration"] * dcost_angular_acceleration +
                      self.factors["curvature"] * dcost_curvature)

        return self._cost

    def cost_difference_resampled(self, trajectory1: Trajectory, trajectory2: Trajectory, goal: Goal) -> float:
        """ Calculate the sum of the cost elements differences between two trajectories given a goal,
        at sampled points along the pathlength

        Args:
            trajectory1: The first trajectory to examine
            trajectory2: The second trajectory to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost difference value
        """

        trajectories_resampled = []
        for trajectory in [trajectory1, trajectory2]:
            goal_reached = self._goal_reached(trajectory, goal)
            trajectory_resampled = VelocityTrajectory(trajectory.path[:goal_reached],
                                                         trajectory.velocity[:goal_reached],
                                                         trajectory.heading[:goal_reached],
                                                         trajectory.timesteps[:goal_reached])
            trajectories_resampled.append(trajectory_resampled)

        n = min(len(trajectory.velocity) for trajectory in trajectories_resampled)
        # handle edge case where goal is reached straight away
        if n == 0:
            self._cost = 0.
            return self._cost

        trajectories_resampled = [self.resample_trajectory(trajectory, n) for trajectory in trajectories_resampled]

        # handle case when we reach the end of the trajectory
        if len(trajectories_resampled[0].path) == 1 or len(trajectories_resampled[1].path) == 1:
            self._cost = 0.
            return self._cost

        dcost_time_to_goal = self._d_time_to_goal(trajectories_resampled[0], trajectories_resampled[1])
        dcost_velocity = self._d_velocity(trajectories_resampled[0], trajectories_resampled[1])
        dcost_longitudinal_acceleration = self._d_longitudinal_acceleration(trajectories_resampled[0],
                                                                            trajectories_resampled[1])
        dcost_longitudinal_jerk = self._d_longitudinal_jerk(trajectories_resampled[0], trajectories_resampled[1])
        dcost_heading = self._d_heading(trajectories_resampled[0], trajectories_resampled[1])
        dcost_angular_velocity = self._d_angular_velocity(trajectories_resampled[0], trajectories_resampled[1])
        dcost_angular_acceleration = self._d_angular_acceleration(trajectories_resampled[0], trajectories_resampled[1])
        dcost_curvature = self._d_curvature(trajectories_resampled[0], trajectories_resampled[1])

        self._cost = (self.factors["time"] * dcost_time_to_goal +
                      self.factors["velocity"] * dcost_velocity +
                      self.factors["acceleration"] * dcost_longitudinal_acceleration +
                      self.factors["jerk"] * dcost_longitudinal_jerk +
                      self.factors["heading"] * dcost_heading +
                      self.factors["angular_velocity"] * dcost_angular_velocity +
                      self.factors["angular_acceleration"] * dcost_angular_acceleration +
                      self.factors["curvature"] * dcost_curvature)

        return self._cost

    def resample_trajectory(self, trajectory: VelocityTrajectory, n: int, k: int = 3):
        zeros = [id for id in np.argwhere(trajectory.velocity <= trajectory.velocity_stop)]
        if zeros and len(zeros) < len(trajectory.velocity) - 1:
            path = np.delete(trajectory.path, zeros, axis=0)
            velocity = np.delete(trajectory.velocity, zeros)
            heading = np.delete(trajectory.heading, zeros)
            timesteps = np.delete(trajectory.timesteps, zeros)
        else:
            path = trajectory.path
            velocity = trajectory.velocity
            heading = trajectory.heading
            timesteps = trajectory.timesteps

        trajectory_nostop = VelocityTrajectory(path, velocity, heading, timesteps)
        u = trajectory_nostop.pathlength
        u_new = linspace(0, u[-1], n)
        k = min(k, len(velocity) - 1)

        if k != 0:
            tck, _ = splprep([path[:, 0], path[:, 1], velocity, heading], u=u, k=k, s=0)
            tck[0] = self.fix_points(tck[0])
            path_new = np.empty((n, 2), float)
            path_new[:, 0], path_new[:, 1], velocity_new, heading_new = splev(u_new, tck)
            trajectory_resampled = VelocityTrajectory(path_new, velocity_new, heading_new)
        else:
            trajectory_resampled = trajectory

        return trajectory_resampled

    def fix_points(self, points, eps=1e-4):
        idx = np.argwhere(np.diff(points) == 0).T.tolist()[0]
        groups = [list(g) for g in mit.consecutive_groups(idx)]
        for g in groups:
            if g[0] == 0:
                continue
            glen = len(g)
            for i, ix in enumerate(g):
                points[ix] -= (glen - i) * eps
        return points

    def _goal_reached(self, trajectory: Trajectory, goal: Goal) -> int:
        """ Returns the trajectory index at which the goal is reached.
        If the goal is never reached, throws an Error """
        for i, p in enumerate(trajectory.path):
            if goal.reached(p):
                return i

        raise IndexError("Iterated through trajectory without reaching goal")

    def _time_to_goal(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        return trajectory.times[goal_reached_i]

    def _velocity(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = trajectory.velocity[:goal_reached_i]
        limit = self._limits["velocity"]
        cost = np.clip(cost, 0, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], cost)

    def _longitudinal_acceleration(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = trajectory.acceleration[:goal_reached_i]
        limit = self._limits["acceleration"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], np.abs(cost))

    def _longitudinal_jerk(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = trajectory.jerk[:goal_reached_i]
        limit = self._limits["jerk"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], np.abs(cost))

    def _heading(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = np.unwrap(trajectory.heading[:goal_reached_i]) % (2 * np.pi)
        limit = self._limits["heading"]
        cost = cost / limit  # no clipping here because heading is unwrapped
        return np.dot(trajectory.timesteps[:goal_reached_i], np.abs(cost))

    def _angular_velocity(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = trajectory.angular_velocity[:goal_reached_i]
        limit = self._limits["angular_velocity"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], np.abs(cost))

    def _angular_acceleration(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = trajectory.angular_acceleration[:goal_reached_i]
        limit = self._limits["angular_acceleration"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], np.abs(cost))

    def _curvature(self, trajectory: Trajectory, goal_reached_i: int) -> float:
        cost = trajectory.curvature[:goal_reached_i]
        limit = self._limits["curvature"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], np.abs(cost))

    def _safety(self) -> float:
        raise NotImplementedError

    def _d_time_to_goal(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        return abs(trajectory1.duration - trajectory2.duration)

    def _d_velocity(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["velocity"]
        dcost = np.abs(
            np.clip(trajectory1.velocity, 0, limit) / limit - np.clip(trajectory2.velocity, 0, limit) / limit)
        return dcost.mean()

    def _d_longitudinal_acceleration(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["acceleration"]
        dcost = np.abs(
            np.clip(trajectory1.acceleration, -limit, limit) / limit - np.clip(trajectory2.acceleration, -limit,
                                                                               limit) / limit)
        return dcost.mean()

    def _d_longitudinal_jerk(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["jerk"]
        dcost = np.abs(
            np.clip(trajectory1.jerk, -limit, limit) / limit - np.clip(trajectory2.jerk, -limit, limit) / limit)
        return dcost.mean()

    def _d_heading(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["heading"]
        # no clipping because heading is unwrapped.
        dcost = np.abs(np.unwrap(trajectory1.heading) / limit - np.unwrap(trajectory2.heading) / limit)
        return dcost.mean()

    def _d_angular_velocity(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["angular_velocity"]
        dcost = np.abs(
            np.clip(trajectory1.angular_velocity, -limit, limit) / limit - np.clip(trajectory2.angular_velocity, -limit,
                                                                                   limit) / limit)
        return dcost.mean()

    def _d_angular_acceleration(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["angular_acceleration"]
        dcost = np.abs(
            np.clip(trajectory1.angular_acceleration, -limit, limit) / limit - np.clip(trajectory2.angular_acceleration,
                                                                                       -limit, limit) / limit)
        return dcost.mean()

    def _d_curvature(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        limit = self._limits["curvature"]
        dcost = np.abs(np.clip(trajectory1.curvature, -limit, limit) / limit - np.clip(trajectory2.curvature, -limit,
                                                                                       limit) / limit)
        return dcost.mean()

    def _d_safety(self, trajectory1: Trajectory, trajectory2: Trajectory) -> float:
        raise NotImplementedError

    @property
    def factors(self) -> Dict[str, float]:
        """Returns a dictionary of the cost factors."""
        return self._factors

    @property
    def limits(self) -> Dict[str, float]:
        """Returns a dictionary of the cost quantities' absolute limits."""
        return self._limits

    @property
    def cost(self) -> Optional[float]:
        """ The cost from the latest trajectory cost calculation call."""
        return self._cost

    @property
    def cost_components(self) -> Optional[Dict[str, float]]:
        """ Return a dictionary of cost components that were calculated with the last call to trajectory_cost()"""
        return self._components
