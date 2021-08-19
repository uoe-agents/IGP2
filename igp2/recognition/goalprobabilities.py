import random
from copy import copy
from typing import List, Dict

import numpy as np

from igp2.goal import Goal
from igp2.trajectory import VelocityTrajectory


class GoalWithType:
    """Tuple of a Goal object and a goal_type string defining an action the vehicle has to perform to
     reach the goal (e.g. turn left). In the current implementation, goal_types are not defined."""

    def __init__(self):
        print("never called in this case")

    def __new__(cls, goal: Goal = None, goal_type: str = None):
        return goal, goal_type


class GoalsProbabilities:
    """Class used to store and update goal probabilities, as well as store useful results such as the priors,
     likelihoods, generated trajectories and rewards """

    def __init__(self, goals: List[Goal] = None, goal_types: List[List[str]] = None, priors: List[float] = None):
        """Creates a new GoalsProbabilities object.

        Args:
            goals: a list of goals objects representing the scenarios goals.
            goal_types: optionally, refine the goals with goal types.
            priors: optionally, a list of goal priors measured from the dataset.
                    If unused, the priors will be set to a uniform distribution.
        """
        self._goals_and_types = []
        if goal_types is None:
            for goal in goals:
                goal_and_type = GoalWithType(goal, None)
                self._goals_and_types.append(goal_and_type)
        else:
            for goal, goal_type_arr in zip(goals, goal_types):
                for goal_type in goal_type_arr:
                    goal_and_type = GoalWithType(goal, goal_type)
                    self._goals_and_types.append(goal_and_type)

        if priors is None:
            self._goals_priors = dict.fromkeys(self._goals_and_types, self.uniform_distribution())
        else:
            self._goals_priors = dict(zip(self._goals_and_types, priors))

        # Actual normalised goal and trajectories probabilities
        self._goals_probabilities = copy(self._goals_priors)
        self._trajectories_probabilities = dict.fromkeys(self._goals_and_types, [])

        # To store trajectory data
        self._optimum_trajectory = dict.fromkeys(self._goals_and_types, None)
        self._current_trajectory = copy(self._optimum_trajectory)
        self._all_trajectories = {key: [] for key in self._goals_and_types}

        # Reward data
        self._optimum_reward = copy(self._optimum_trajectory)
        self._current_reward = copy(self._optimum_trajectory)
        self._all_rewards = {key: [] for key in self._goals_and_types}

        # Reward difference data
        self._reward_difference = copy(self._optimum_trajectory)
        self._all_reward_differences = {key: [] for key in self._goals_and_types}

        # The goal likelihoods
        self._likelihood = copy(self._optimum_trajectory)

    def uniform_distribution(self) -> float:
        """Generates a uniform distribution across each GoalWithType object"""
        return float(1 / len(self._goals_and_types))

    def sample_goals(self, k: int = 1) -> List[GoalWithType]:
        """Used to randomly sample a goal according to the goals probability distribution."""
        goals = list(self.goals_probabilities.keys())
        weights = self.goals_probabilities.values()
        return random.choices(goals, weights=weights, k=k)

    def sample_trajectories_to_goal(self, goal: GoalWithType, k: int = 1) -> List[VelocityTrajectory]:
        """ Randomly sample up to k trajectories from all_trajectories to the given goal
         using the trajectory distributions"""
        assert goal in self.trajectories_probabilities, f"Goal {goal} not in trajectories_probabilities!"
        assert goal in self.all_trajectories, f"Goal {goal} not in all_trajectories!"

        trajectories = self._all_trajectories[goal]
        weights = self._trajectories_probabilities[goal]
        return random.choices(trajectories, weights=weights, k=k)

    @property
    def goals_probabilities(self) -> Dict[GoalWithType, float]:
        """Returns the current goals probabilities."""
        return self._goals_probabilities

    @property
    def goals_priors(self) -> Dict[GoalWithType, float]:
        """Return the goals priors."""
        return self._goals_priors

    @property
    def trajectories_probabilities(self) -> Dict[GoalWithType, List[float]]:
        """ Return the trajectories probability distribution to each goal"""
        return self._trajectories_probabilities

    @property
    def optimum_trajectory(self) -> Dict[GoalWithType, VelocityTrajectory]:
        """Returns the trajectory from initial vehicle position generated to each goal to calculate the likelihood."""
        return self._optimum_trajectory

    @property
    def current_trajectory(self) -> Dict[GoalWithType, VelocityTrajectory]:
        """Returns the real vehicle trajectory, extended by the trajectory
         from current vehicle position that was generated to each goal to calculate the likelihood."""
        return self._current_trajectory

    @property
    def all_trajectories(self) -> Dict[GoalWithType, List[VelocityTrajectory]]:
        """ Returns the real vehicle trajectory, extended by all possible generated paths to a given goal."""
        return self._all_trajectories

    @property
    def optimum_reward(self) -> Dict[GoalWithType, float]:
        """Returns the reward generated by the optimum_trajectory for each goal"""
        return self._optimum_reward

    @property
    def current_reward(self) -> Dict[GoalWithType, float]:
        """Returns the reward generated by the current_trajectory for each goal"""
        return self._current_reward

    @property
    def all_rewards(self) -> Dict[GoalWithType, List[float]]:
        """Returns a list of rewards generated by all_trajectories for each goal"""
        return self._all_rewards

    @property
    def reward_difference(self) -> Dict[GoalWithType, float]:
        """Returns the reward generated by the optimum_trajectory for each goal,
        if we are not using the reward_as_difference toggle."""
        return self._reward_difference

    @property
    def all_reward_differences(self) -> Dict[GoalWithType, List[float]]:
        """Returns the rewards generated by all_trajectories for each goal,
        if we are using the reward_as_difference toggle."""
        return self._all_reward_differences

    @property
    def likelihood(self) -> Dict[GoalWithType, float]:
        """Returns the computed likelihoods for each goal"""
        return self._likelihood

    @property
    def goals_and_types(self) -> List[GoalWithType]:
        """ Return each goal and the possible corresponding type """
        return self._goals_and_types