from copy import copy
from typing import Dict, List
import logging
import numpy as np

from igp2.core.trajectory import StateTrajectory
from igp2.core.goal import Goal
from igp2.core.cost import Cost

logger = logging.getLogger(__name__)


class Reward:
    """ Used to calculate rollout rewards in MCTS. """

    def __init__(self,
                 time_discount: float = 0.99,
                 factors: Dict[str, float] = None,
                 default_rewards: Dict[str, float] = None):
        """ Initialise a new Reward class.

        Args:
            time_discount: Discounting factor for each time step.
            factors: Reward component factors.
            default_rewards: Default rewards for specific events during rollout.
        """
        self._factors = {
            "time": 1.0,
            "jerk": -0.1,
            "angular_velocity": -0.1,
            "curvature": -0.1,
            "coll": 1.,
            "term": 1.,
            "dead": 1.,
        }
        if factors is not None:
            self._factors.update(factors)

        self._default_rewards = {
            "coll": -1.,
            "term": -1.,
            "dead": -1.,
        }
        if default_rewards is not None:
            self._default_rewards.update(default_rewards)

        self.COMPONENTS = list(set(self._factors).union(set(self._default_rewards)))

        self._time_discount = time_discount
        self._components = None
        self._cost_components = None
        self._reward = None
        self.reset()

    def __call__(self, *args, **kwargs):
        return self._calc_reward(*args, **kwargs)

    def _calc_reward(self,
                     collisions: List["Agent"] = None,
                     alive: bool = True,
                     ego_trajectory: StateTrajectory = None,
                     goal: Goal = None,
                     depth_reached: bool = False
                     ) -> float:
        if collisions:
            self._reward = self._factors.get("coll", 1.) * self._default_rewards.get("coll", -1.)
            self._components["coll"] = self._reward
        elif not alive:
            self._reward = self._factors.get("dead", 1.) * self._default_rewards.get("dead", -1.)
            self._components["dead"] = self._reward
        elif ego_trajectory is not None and goal is not None:
            trajectory_costs = self.trajectory_reward(ego_trajectory, goal)
            trajectory_rewards = {comp: self._factors[comp] * rew for comp, rew in trajectory_costs.items()}
            self._reward = sum(trajectory_rewards.values())
            self._cost_components = trajectory_costs
            self._components.update(trajectory_rewards)
        elif depth_reached:
            self._reward = self._factors.get("term", 1.) * self._default_rewards.get("term", -1.)
            self._components["term"] = self._reward

        return self._reward

    def trajectory_reward(self, trajectory: StateTrajectory, goal: Goal) -> Dict[str, float]:
        """ Calculate reward components for a given trajectory. """
        costs = Cost()
        costs.trajectory_cost(trajectory, goal)

        return {
            "time": self._time_discount ** trajectory.duration,
            "jerk": costs.cost_components["jerk"],
            "angular_velocity": costs.cost_components["angular_velocity"],
            "curvature": costs.cost_components["curvature"]
        }

    def reset(self):
        """ Reset rewards to initialisation values. """
        self._reward = None
        self._components = {rew: None for rew in self.COMPONENTS}

    @property
    def default_rewards(self) -> Dict[str, float]:
        """ Default rewards for rollout events e.g. collision, termination, and death. """
        return self._default_rewards

    @property
    def reward(self) -> float:
        """ The last calculate reward. """
        return self._reward

    @property
    def reward_components(self) -> Dict[str, float]:
        """ The current reward components """
        return self._components

    @property
    def cost_components(self) -> Dict[str, float]:
        """ The trajectory cost components. """
        # cost_components = copy(self._components)
        # if self._components["time"] is not None:
        #     tc = cost_components["time"]
        #     cost_components["time"] = np.log(tc) / np.log(self._time_discount)
        return self._cost_components

    @property
    def time_discount(self) -> float:
        """ Discounting factor for time-to-goal reward component. """
        return self._time_discount

    @property
    def factors(self) -> Dict[str, float]:
        """ Reward component factors. """
        return self._factors
