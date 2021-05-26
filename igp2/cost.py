import numpy as np
from igp2.trajectory import Trajectory
from igp2.goal import Goal
from shapely.geometry import Point


class Cost:
    """ Define the exact cost signal of a trajectory.
    The IGP2 paper refers to this as reward, which can be interpreted as negative cost. """
    def __init__(self, w_time: float = 1., w_acc: float = 1., w_jerk: float = 1., w_angvel: float = 1.,
                 w_angacc: float = 1., w_curv: float = 1., w_saf: float = 1.):
        """ Initialise a new Cost class with the given weights.

        Args:
            w_time: Time to goal weight
            w_acc: Acceleration weight
            w_jerk: Jerk weight
            w_angvel: Angular velocity weight
            w_angacc: Angular acceleration weight
            w_curv: Curvature weight
            w_saf: Safety weight
        """

        self._factors = {"time": w_time, "acc": w_acc, "jerk": w_jerk, "angvel": w_angvel,
                         "angacc": w_angacc, "curv": w_curv, "saf": w_saf}

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
                      self.factors["acc"] * self._longitudinal_acceleration() +
                      self.factors["jerk"] * self._longitudinal_jerk() +
                      self.factors["angvel"] * self._angular_velocity() +
                      self.factors["angacc"] * self._angular_acceleration() +
                      self.factors["curv"] * self._curvature())

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
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i],
                      np.abs(self._trajectory.acceleration[:self._goal_reached_i]))

    def _longitudinal_jerk(self) -> float:
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i],
                      np.abs(self._trajectory.jerk[:self._goal_reached_i]))

    def _angular_velocity(self) -> float:
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i],
                      np.abs(self._trajectory.angular_velocity[:self._goal_reached_i]))

    def _angular_acceleration(self) -> float:
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i],
                      np.abs(self._trajectory.angular_acceleration[:self._goal_reached_i]))

    def _curvature(self) -> float:
        return np.dot(self._trajectory.trajectory_dt()[:self._goal_reached_i], np.abs(self._trajectory.curvature[:self._goal_reached_i]))

    def _safety(self) -> float:
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        return self._factors

    @property
    def goal(self) -> Goal:
        return self._goal
