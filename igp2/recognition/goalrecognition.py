from igp2.opendrive.map import Map
import numpy as np
import random
import math
import logging
import copy
from typing import Callable, List, Dict, Tuple

from igp2.agent import AgentState, TrajectoryAgent
from igp2.cost import Cost
from igp2.goal import Goal, PointGoal
from igp2.trajectory import *
from igp2.planlibrary.maneuver import Maneuver
from igp2.recognition.astar import AStar

logger = logging.getLogger(__name__)

class GoalWithType:

    def __init__(self):
        print ("never called in this case")

    def __new__(cls, goal : Goal = None, goal_type : str = None):
        return (goal, goal_type)

class GoalsProbabilities:

    def __init__(self, goals : List[Goal] = None, goal_types : List[List[str]] = None):
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

        self._goals_probabilities = dict.fromkeys(self._goals_and_types, self.uniform_distribution())
        self._goals_priors = copy.copy(self._goals_probabilities)
        self._optimum_trajectory = dict.fromkeys(self._goals_and_types, None)
        self._current_trajectory = copy.copy(self._optimum_trajectory)

    def uniform_distribution(self) -> float:
        return float(1/len(self._goals_and_types))

    def sample(self, k : int = 1) -> List[GoalWithType]:
        return random.choices(list(self.goals_probabilities.keys()), weights=self.goals_probabilities.values(), k=k)

    @property
    def goals_probabilities(self) -> Dict[GoalWithType, float]:
        return self._goals_probabilities

    @property
    def goals_priors(self) -> Dict[GoalWithType, float]:
        return self._goals_priors

    @property
    def optimum_trajectory(self) -> Dict[GoalWithType, VelocityTrajectory]:
        return self._optimum_trajectory

    @property
    def current_trajectory(self) -> Dict[GoalWithType, VelocityTrajectory]:
        return self._current_trajectory

    @property
    def optimum_reward(self) -> Dict[GoalWithType, float]:
        raise NotImplementedError

    @property
    def current_reward(self) -> Dict[GoalWithType, float]:
        raise NotImplementedError

    @property
    def likelihood(self) -> Dict[GoalWithType, float]:
        raise NotImplementedError

class GoalRecognition:

    def __init__(self, astar: AStar, smoother: VelocitySmoother, scenario_map: Map, cost: Cost = None, beta: float = 1.):
        self._beta = beta
        self._astar = astar
        self._smoother = smoother
        self._cost = Cost() if cost is None else cost
        self._scenario_map = scenario_map

    def update_goals_probabilities(self, goals_probabilities: GoalsProbabilities, 
    trajectory: StateTrajectory, agentId: int, frame_ini: Dict[int, AgentState], 
    frame: Dict[int, AgentState], maneuver: Maneuver = None) -> GoalsProbabilities :
        norm_factor = 0.
        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            try:
                goal = goal_and_type[0]
                #4. and 5. Generate optimum trajectory from initial point and smooth it
                if goals_probabilities.optimum_trajectory[goal_and_type] == None:
                    logger.debug("Generating optimum trajectory")
                    goals_probabilities.optimum_trajectory[goal_and_type] = self.generate_trajectory(agentId, frame_ini, goal)
                opt_trajectory = goals_probabilities.optimum_trajectory[goal_and_type]
                #7. and 8. Generate optimum trajectory from last observed point and smooth it
                if goals_probabilities.current_trajectory[goal_and_type] == None:
                    current_trajectory = opt_trajectory
                else:
                    current_trajectory = self.generate_trajectory(agentId, frame, goal, maneuver)
                    #10. current_trajectory = join(trajectory, togoal_trajectory)
                    current_trajectory.insert(trajectory)
                goals_probabilities.current_trajectory[goal_and_type] = current_trajectory
                #6,9,10. calculate likelihood, update goal probabilities
                likelihood = self.likelihood(current_trajectory, opt_trajectory, goal)
            except RuntimeError as e:
                logger.debug(str(e))
                likelihood = 0.
            goals_probabilities.goals_probabilities[goal_and_type] = goals_probabilities.goals_priors[goal_and_type] * likelihood
            norm_factor += likelihood * goals_probabilities.goals_priors[goal_and_type]

        # then divide prob by norm_factor to normalise
        for key, prob in goals_probabilities.goals_probabilities.items():
            try:
                goals_probabilities.goals_probabilities[key] = prob / norm_factor
            except ZeroDivisionError as e:
                logger.debug("All goals unreacheable. Setting all probabilities to 0.")
                break

    def generate_trajectory(self, agentId: int, frame: Dict[int, AgentState], goal: Goal, maneuver: Maneuver = None) -> VelocityTrajectory:
        trajectories, _ = self._astar.search(agentId, frame, goal, self._scenario_map, maneuver)
        if len(trajectories) == 0 : raise RuntimeError("Goal is unreachable")
        trajectory = trajectories[0]
        self._smoother.load_trajectory(trajectory)
        trajectory.velocity = self._smoother.split_smooth()
        return trajectory

    def likelihood(self, current_trajectory : VelocityTrajectory, opt_trajectory: VelocityTrajectory, goal: Goal) -> float :
        r_current = self.reward(current_trajectory, goal)
        r_opt = self.reward(opt_trajectory, goal)
        return math.exp(self._beta * (r_current - r_opt))

    def reward(self, trajectory: Trajectory, goal: Goal) -> float:
        return - self._cost.trajectory_cost(trajectory, goal)