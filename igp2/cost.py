import numpy as np
from trajectory import VelocityTrajectory

class Cost:

    def __init__(self, w_acc: float = 1., w_jerk: float = 1., w_angvel: float = 1.,
                    w_angacc: float = 1., w_curv: float = 1., w_saf: float = 1.):

        self._factors = {"acc": w_acc, "jerk": w_jerk, "angvel": w_angvel,
                        "angacc": w_angacc, "curv": w_curv, "saf": w_saf}

        self._goal = None
    
    def trajectory_cost(self, trajectory: VelocityTrajectory) -> float:
        raise NotImplementedError

    def time_to_goal(self) -> float:
        raise NotImplementedError
    
    def longitudinal_acceleration(self) -> float:
        raise NotImplementedError

    def longitudinal_jerk(self) -> float:
        raise NotImplementedError

    def angular_velocity(self) -> float:
        raise NotImplementedError

    def angular_acceleration(self) -> float:
        raise NotImplementedError

    def curvature(self) -> float:
        raise NotImplementedError

    def safety(self) -> float:
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        return self._factors

    @property
    def goal(self) -> None:
        return self.goal