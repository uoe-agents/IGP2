import numpy as np
from typing import Dict

from numpy.core.numeric import NaN

from igp2.trajectory import Trajectory, VelocityTrajectory
from igp2.goal import Goal
from shapely.geometry import Point


class Cost:
    """ Define the exact cost signal of a trajectory.
    The IGP2 paper refers to this as reward, which can be interpreted as negative cost. """
    def __init__(self, factors: Dict[str, float] = None, limits: Dict[str, float] = None):
        """ Initialise a new Cost class with the given weights.

        Args:
            factors: dictionary of weights for different costs, comprising of
                w_time: Time to goal weight
                w_acc: Acceleration weight
                w_jerk: Jerk weight
                w_angvel: Angular velocity weight
                w_angacc: Angular acceleration weight
                w_curv: Curvature weight
                w_saf: Safety weight (not implemented)
            limits: dictionary of limits for different costs, comprising of

        """

        self._factors = {"time": 1., "acceleration": 1., "jerk": 1., "angular_velocity": 1.,
                         "angular_acceleration": 1., "curvature": 1., "safety": 1.} if factors is None else factors

        self._limits = {"acceleration": 4.3442, "jerk": 125.63, "angular_velocity": 1.013,
                         "angular_acceleration": 35.127, "curvature": 108.04} if limits is None else limits

        self._cost = None
        self._dcost = None

    def trajectory_cost(self, trajectory: Trajectory, goal: Goal) -> float:
        """ Calculate the total cost of the trajectory given a goal.

        Args:
            trajectory: The trajectory to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost value
        """
        goal_reached_i = self._goal_reached(trajectory, goal)

        self._cost =( self.factors["time"] * abs(self._time_to_goal(trajectory, goal_reached_i)) +
                      self.factors["acceleration"] * abs(self._longitudinal_acceleration(trajectory, goal_reached_i)) +
                      self.factors["jerk"] * abs(self._longitudinal_jerk(trajectory, goal_reached_i)) +
                      self.factors["angular_velocity"] * abs(self._angular_velocity(trajectory, goal_reached_i)) +
                      self.factors["angular_acceleration"] * abs(self._angular_acceleration(trajectory, goal_reached_i)) +
                      self.factors["curvature"] * abs(self._curvature(trajectory, goal_reached_i)) )

        return self._cost

    def cost_difference(self, trajectory1: Trajectory, trajectory2: Trajectory, goal: Goal) -> float:
        """ Calculate the sum of the cost elements differences between two trajectories, given a goal.

        Args:
            trajectory1, trajectory 2: The trajectories to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost difference value
        """
        goal_reached_i1 = self._goal_reached(trajectory1, goal)
        goal_reached_i2 = self._goal_reached(trajectory2, goal)

        dcost_time_to_goal = abs(self._time_to_goal(trajectory1, goal_reached_i1) - self._time_to_goal(trajectory2, goal_reached_i2))
        dcost_longitudinal_acceleration = abs(self._longitudinal_acceleration(trajectory1, goal_reached_i1) - self._longitudinal_acceleration(trajectory2, goal_reached_i2))
        dcost_longitudinal_jerk = abs(self._longitudinal_jerk(trajectory1, goal_reached_i1) - self._longitudinal_jerk(trajectory2, goal_reached_i2))
        dcost_angular_velocity = abs(self._angular_velocity(trajectory1, goal_reached_i1) - self._angular_velocity(trajectory2, goal_reached_i2))
        dcost_angular_acceleration = abs(self._angular_acceleration(trajectory1, goal_reached_i1) - self._angular_acceleration(trajectory2, goal_reached_i2))
        dcost_curvature = abs(self._curvature(trajectory1, goal_reached_i1) - self._curvature(trajectory2, goal_reached_i2))

        self._cost = (self.factors["time"] * dcost_time_to_goal +
                self.factors["acceleration"] * dcost_longitudinal_acceleration +
                self.factors["jerk"] * dcost_longitudinal_jerk +
                self.factors["angular_velocity"] * dcost_angular_velocity +
                self.factors["angular_acceleration"] * dcost_angular_acceleration +
                self.factors["curvature"] * dcost_curvature)

        return self._cost

    def cost_difference_resampled(self, trajectory1: Trajectory, trajectory2: Trajectory, goal: Goal) -> float:
        """ Calculate the sum of the cost elements differences between two trajectories given a goal, at sampled points along the pathlength

        Args:
            trajectory1, trajectory 2: The trajectories to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost difference value
        """

        goal_reached_i1 = self._goal_reached(trajectory1, goal)
        goal_reached_i2 = self._goal_reached(trajectory2, goal)

        trajectories_resampled = []
        for trajectory in [trajectory1, trajectory2]:
            goal_reached = self._goal_reached(trajectory, goal)
            trajectory_resampled = VelocityTrajectory(trajectory.path[:goal_reached], trajectory.velocity[:goal_reached],
                trajectory.heading[:goal_reached], trajectory.timesteps[:goal_reached])
            trajectories_resampled.append(trajectory_resampled)
        
        n = min(len(trajectory) for trajectory in trajectories_resampled)
        for trajectory in trajectories_resampled:
            trajectory = self.resample_trajectory(trajectory, n)

        dcost_time_to_goal = self._d_time_to_goal(trajectories_resampled[0], trajectories_resampled[1])
        dcost_longitudinal_acceleration = self._d_longitudinal_acceleration(trajectories_resampled[0], trajectories_resampled[1])
        dcost_longitudinal_jerk = self._d_longitudinal_jerk(trajectories_resampled[0], trajectories_resampled[1])
        dcost_angular_velocity = self._d_angular_velocity(trajectories_resampled[0], trajectories_resampled[1])
        dcost_angular_acceleration = self._d_angular_acceleration(trajectories_resampled[0], trajectories_resampled[1])
        dcost_curvature = self._d_curvature(trajectories_resampled[0], trajectories_resampled[1])

        self._cost = (self.factors["time"] * dcost_time_to_goal +
                self.factors["acceleration"] * dcost_longitudinal_acceleration +
                self.factors["jerk"] * dcost_longitudinal_jerk +
                self.factors["angular_velocity"] * dcost_angular_velocity +
                self.factors["angular_acceleration"] * dcost_angular_acceleration +
                self.factors["curvature"] * dcost_curvature)

    def resample_trajectory(self, trajectory: Trajectory, n: int):

        return trajectory

    def _goal_reached(self, trajectory: Trajectory, goal: Goal) -> int:
        """ Returns the trajectory index at which the goal is reached.
        If the goal is never reached, throws an Error """
        for i, p in enumerate(trajectory.path):
            p = Point(p)
            if goal.reached(p):
                return i

        raise IndexError("Iterated through trajectory without reaching goal")

    def _time_to_goal(self, trajectory : Trajectory, goal_reached_i : int) -> float:
        return trajectory.times[goal_reached_i]

    def _longitudinal_acceleration(self, trajectory : Trajectory, goal_reached_i : int) -> float:
        cost = trajectory.acceleration[:goal_reached_i]
        limit = self._limits["acceleration"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], cost)

    def _longitudinal_jerk(self, trajectory : Trajectory, goal_reached_i : int) -> float:
        cost = trajectory.jerk[:goal_reached_i]
        limit = self._limits["jerk"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], cost)

    def _angular_velocity(self, trajectory : Trajectory, goal_reached_i : int) -> float:
        cost = trajectory.angular_velocity[:goal_reached_i]
        limit = self._limits["angular_velocity"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], cost)

    def _angular_acceleration(self, trajectory : Trajectory, goal_reached_i : int) -> float:
        cost = trajectory.angular_acceleration[:goal_reached_i]
        limit = self._limits["angular_acceleration"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], cost)

    def _curvature(self, trajectory : Trajectory, goal_reached_i : int) -> float:
        cost = trajectory.curvature[:goal_reached_i]
        limit = self._limits["curvature"]
        cost = np.clip(cost, -limit, limit) / limit
        return np.dot(trajectory.timesteps[:goal_reached_i], cost)

    def _safety(self) -> float:
        raise NotImplementedError

    def _d_time_to_goal(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    def _d_longitudinal_acceleration(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    def _d_longitudinal_jerk(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    def _d_angular_velocity(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    def _d_angular_acceleration(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    def _d_curvature(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    def _d_safety(self, trajectory1 : Trajectory, trajectory2 : Trajectory) -> float:
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        return self._factors
