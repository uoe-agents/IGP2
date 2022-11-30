from typing import Dict, List
import logging

import igp2 as ip

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
            "curvature": -0.1
        } if factors is None else factors

        self._default_rewards = {
            "coll": -1,
            "term": -1,
            "dead": -1,
        } if default_rewards is None else default_rewards

        self.COMPONENTS = list(self._factors) + list(self._default_rewards)

        self._time_discount = time_discount
        self._components = None
        self._reward = None
        self.reset()

    def __call__(self, *args, **kwargs):
        return self._calc_reward(*args, **kwargs)

    def _calc_reward(self,
                     collisions: List[ip.Agent] = None,
                     alive: bool = True,
                     ego_trajectory: ip.StateTrajectory = None,
                     goal: ip.Goal = None,
                     depth_reached: bool = False
                     ) -> float:
        if collisions:
            self._reward = self._default_rewards["coll"]
            self._components["coll"] = self._reward
            logger.debug(f"Ego agent collided with agent(s): {collisions}")
        elif not alive:
            self._reward = self._default_rewards["dead"]
            self._components["dead"] = self._reward
            logger.debug(f"Ego died during rollout!")
        elif ego_trajectory is not None and goal is not None:
            trajectory_rewards = self.trajectory_reward(ego_trajectory, goal)
            self._reward = sum([self._factors[comp] * rew for comp, rew in trajectory_rewards.items()])
            self._components.update(trajectory_rewards)
            logger.debug(f"Goal reached!")
        elif depth_reached:
            self._reward = self._default_rewards["term"]
            self._components["term"] = self._reward
            logger.debug("Reached final rollout depth!")

        return self._reward

    def trajectory_reward(self, trajectory: ip.StateTrajectory, goal: ip.Goal) -> Dict[str, float]:
        """ Calculate reward components for a given trajectory. """
        costs = ip.Cost()
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
    def time_discount(self) -> float:
        """ Discounting factor for time-to-goal reward component. """
        return self._time_discount

    @property
    def factors(self) -> Dict[str, float]:
        """ Reward component factors. """
        return self._factors
