import numpy as np
from igp2.trajectory import VelocityTrajectory
from igp2.goal import Goal
from igp2.util import get_curvature
from shapely.geometry import Point

class Cost:

    def __init__(self, goal: Goal, w_time: float = 1., w_acc: float = 1., w_jerk: float = 1., w_angvel: float = 1.,
                    w_angacc: float = 1., w_curv: float = 1., w_saf: float = 1.):

        self._factors = {"time": w_time,"acc": w_acc, "jerk": w_jerk, "angvel": w_angvel,
                        "angacc": w_angacc, "curv": w_curv, "saf": w_saf}

        self._goal = goal
    
    def trajectory_cost(self, trajectory: VelocityTrajectory) -> float:
        self._traj = trajectory
        self._goal_reached_i = self._goal_reached()

        self._cost = self.factors["time"] * self._time_to_goal() + \
                     self.factors["acc"] * self._longitudinal_acceleration() + \
                     self.factors["jerk"] * self._longitudinal_jerk() + \
                     self.factors["angvel"] * self._angular_velocity() + \
                     self.factors["angacc"] * self._angular_acceleration() + \
                     self.factors["curv"] * self._curvature()

        return self._cost

    def _goal_reached(self) -> int:
        """Returns the trajectory indice at which the goal is reached.
        If the goal is never reached, throws an Error"""
        for i, p in enumerate(self._traj.path):
            p = Point(p)
            if self._goal.reached(p): return i
        
        raise IndexError("Iterated through trajectory without reaching goal")

    def _time_to_goal(self) -> float:
        return self._traj.trajectory_times()[self._goal_reached_i]
    
    def _longitudinal_acceleration(self) -> float:
        return np.average(np.abs(self._traj.acceleration[:self._goal_reached_i]))

    def _longitudinal_jerk(self) -> float:
        return np.average(np.abs(self._traj.jerk[:self._goal_reached_i]))

    def _angular_velocity(self) -> float:
        return np.average(np.abs(self._traj.angular_velocity[:self._goal_reached_i]))

    def _angular_acceleration(self) -> float:
        return np.average(np.abs(self._traj.angular_acceleration[:self._goal_reached_i]))

    def _curvature(self) -> float:
        return np.average(np.abs(get_curvature(self._traj.path[:self._goal_reached_i])))

    def _safety(self) -> float:
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        return self._factors

    @property
    def goal(self) -> Goal:
        return self.goal