import numpy as np
from typing import Dict

from numpy.core.numeric import NaN

from igp2.trajectory import Trajectory
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
        self._trajectory = None
        self._goal_reached_i = None

    def trajectory_cost(self, trajectory: Trajectory, goal: Goal) -> float:
        """ Calculate the total cost of the trajectory given a goal.

        Args:
            trajectory: The trajectory to examine
            goal: The goal to reach

        Returns:
            A scalar floating-point cost value
        """
        self._trajectory = trajectory
        self._goal_reached_i = self._goal_reached(goal)

        self._cost = (self.factors["time"] * self._time_to_goal() +
                      self.factors["acceleration"] * self._longitudinal_acceleration() +
                      self.factors["jerk"] * self._longitudinal_jerk() +
                      self.factors["angular_velocity"] * self._angular_velocity() +
                      self.factors["angular_acceleration"] * self._angular_acceleration() +
                      self.factors["curvature"] * self._curvature())

        return self._cost

    def _goal_reached(self, goal) -> int:
        """ Returns the trajectory index at which the goal is reached.
        If the goal is never reached, throws an Error """
        for i, p in enumerate(self._trajectory.path):
            p = Point(p)
            if goal.reached(p):
                return i

        raise IndexError("Iterated through trajectory without reaching goal")

    def _time_to_goal(self) -> float:
        return self._trajectory.trajectory_times()[self._goal_reached_i]

    def _longitudinal_acceleration(self) -> float:
        cost = np.abs(self._trajectory.acceleration[:self._goal_reached_i])
        limit = self._limits["acceleration"]
        cost = np.clip(cost, 0, limit) / limit
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i], cost)

    def _longitudinal_jerk(self) -> float:
        cost = np.abs(self._trajectory.jerk[:self._goal_reached_i])
        limit = self._limits["jerk"]
        cost = np.clip(cost, 0, limit) / limit
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i], cost)

    def _angular_velocity(self) -> float:
        cost = np.abs(self._trajectory.angular_velocity[:self._goal_reached_i])
        limit = self._limits["angular_velocity"]
        cost = np.clip(cost, 0, limit) / limit
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i], cost)

    def _angular_acceleration(self) -> float:
        cost = np.abs(self._trajectory.angular_acceleration[:self._goal_reached_i])
        limit = self._limits["angular_acceleration"]
        cost = np.clip(cost, 0, limit) / limit
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i], cost)

    def _curvature(self) -> float:
        cost = np.abs(self._trajectory.curvature[:self._goal_reached_i])
        limit = self._limits["curvature"]
        cost = np.clip(cost, 0, limit) / limit
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i], cost)

    def _safety(self) -> float:
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        return self._factors

    @property
    def goal(self) -> Goal:
        return self._goal
