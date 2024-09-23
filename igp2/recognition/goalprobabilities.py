import random
import numpy as np
import logging
from copy import copy
from operator import itemgetter
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

from igp2.core.goal import Goal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.core.trajectory import VelocityTrajectory
from igp2.core.cost import Cost
from igp2.planlibrary.macro_action import MacroAction


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

        # To store the plans that generated the trajectories
        self._optimum_plan = dict.fromkeys(self._goals_and_types, None)
        self._all_plans = {key: [] for key in self._goals_and_types}

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

    def sample_trajectories_to_goal(self, goal: GoalWithType, k: int = 1) \
            -> Tuple[List[VelocityTrajectory], List[List[MacroAction]]]:
        """ Randomly sample up to k trajectories from all_trajectories to the given goal
         using the trajectory distributions"""
        assert goal in self.trajectories_probabilities, f"Goal {goal} not in trajectories_probabilities!"
        assert goal in self.all_trajectories, f"Goal {goal} not in all_trajectories!"

        trajectories = self._all_trajectories[goal]
        if trajectories:
            weights = self._trajectories_probabilities[goal]
            trajectories = random.choices(trajectories, weights=weights, k=k)
            plans = [self.trajectory_to_plan(goal, traj) for traj in trajectories]
            return trajectories, plans

    def trajectory_to_plan(self, goal: GoalWithType, trajectory: VelocityTrajectory) -> List[MacroAction]:
        """ Return the plan that generated the trajectory. Not used for optimal trajectories. """
        idx = self.all_trajectories[goal].index(trajectory)
        return self.all_plans[goal][idx]

    def map_prediction(self) -> Tuple[GoalWithType, VelocityTrajectory]:
        """ Return the MAP goal and trajectory prediction for each agent. """
        goal = max(self.goals_probabilities, key=self.goals_probabilities.get)
        trajectory, p_trajectory = \
            max(zip(self.all_trajectories[goal],
                    self.trajectories_probabilities[goal]),
                key=itemgetter(1))
        return goal, trajectory

    def add_smoothing(self, alpha: float = 1., uniform_goals: bool = False):
        """ Perform add-alpha smoothing on the probability distribution in place.

         Args:
             alpha: Additive factor for smoothing.
             uniform_goals: Whether to normalise goal probabilities to uniform distribution,
         """
        n_reachable = sum(map(lambda x: len(x) > 0, self.trajectories_probabilities.values()))

        for goal, trajectory_prob in self.trajectories_probabilities.items():
            trajectory_len = len(trajectory_prob)
            if trajectory_len > 0:
                if uniform_goals:
                    self.goals_probabilities[goal] = 1 / n_reachable
                else:
                    self.goals_probabilities[goal] = (self.goals_probabilities[goal] + alpha) / (1 + n_reachable * alpha)
                self.trajectories_probabilities[goal] = \
                    [(prob + alpha) / (1 + trajectory_len * alpha) for prob in trajectory_prob]

    def plot(self,
             scenario_map: Map = None,
             max_n_trajectories: int = 1,
             cost: Cost = None) -> plt.Axes:
        """ Plot the optimal, and predicted trajectories.

        Args:
            scenario_map: Optional road layout to be plotted as background.
            max_n_trajectories: The maximum number of trajectories to plot for each goal if they exist.
            cost: If given, re-calculate cost factors for plotting
        """

        def plot_trajectory(traj, ax_, cmap, goal_, title=""):
            plot_map(scenario_map, markings=True, ax=ax_)
            path, vel = traj.path, traj.velocity
            ax_.scatter(path[:, 0], path[:, 1], c=vel, cmap=cmap, vmin=-4, vmax=20, s=8)
            if cost is not None:
                cost.trajectory_cost(traj, goal_)
                plt.rc('axes', titlesize=8)
                t = str(cost.cost_components)
                t = t[:len(t) // 2] + "\n" + t[len(t) // 2:]
                ax_.set_title(t)
            else:
                ax_.set_title(title)

        color_map_optimal = plt.cm.get_cmap('Reds')
        color_map = plt.cm.get_cmap('Blues')

        valid_goals = [g for g, ts in self._all_trajectories.items() if len(ts) > 0]
        fig, axes = plt.subplots(len(valid_goals), 2, figsize=(12, 9))

        for gid, goal in enumerate(valid_goals):
            plot_trajectory(self._optimum_trajectory[goal], axes[gid, 0], color_map_optimal, goal[0])
            for tid, trajectory in enumerate(self._all_trajectories[goal][:max_n_trajectories], 1):
                plot_trajectory(trajectory, axes[gid, 1], color_map, goal[0])
        return axes
    
    def log(self, lgr: logging.Logger):
        """ Log the probabilities to the given logger. """
        for key, pg_z in self.goals_probabilities.items():
            if pg_z != 0.0:
                lgr.info(f"{key}: {np.round(pg_z, 3)}")
                for i, (plan, prob) in enumerate(zip(self.all_plans[key], self.trajectories_probabilities[key])):
                    lgr.info(f"\tTrajectory {i}: {np.round(prob, 3)}")
                    lgr.info(f"\t\t{plan}")

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
    def optimum_plan(self) -> Dict[GoalWithType, List[MacroAction]]:
        """ Returns the plan from initial vehicle position generated to each goal."""
        return self._optimum_plan

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
    def all_plans(self) -> Dict[GoalWithType, List[List[MacroAction]]]:
        """ Returns all plans from the most recent vehicle position generated to each goal."""
        return self._all_plans

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
