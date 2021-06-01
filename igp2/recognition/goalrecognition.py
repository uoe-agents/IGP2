from igp2.opendrive.map import Map
import numpy as np
import math
from typing import Callable, List, Dict, Tuple

from igp2.agent import AgentState, TrajectoryAgent
from igp2.cost import Cost
from igp2.goal import Goal, PointGoal
from igp2.trajectory import *
from igp2.planlibrary.maneuver import Maneuver
from igp2.recognition.astar import AStar

class GoalWithType:

    def __init__(self):
        print ("never called in this case")

    def __new__(cls, goal : Goal = None, goal_type : str = None):
        return (goal, goal_type)

class GoalsProbabilities:

    def __init__(self, goals : List[Goal] = None, goal_types : List[List[str]] = None):

        self._goals_and_types = []
        for goal, goal_type_arr in zip(goals, goal_types):
            for goal_type in goal_type_arr:
                goal_and_type = GoalWithType(goal, goal_type)
                self._goals_and_types.append(goal_and_type)

        self._goal_probabilities = dict.fromkeys(self._goals_and_types, self.uniform_distribution())

    def uniform_distribution(self) -> float:
        return float(1/len(self._goals_and_types))

    def sample(self, k : int = 1) -> List[GoalWithType]:
        raise NotImplementedError

    @property
    def goals_probabilities(self) -> dict:
        return self._goal_probabilities

class GoalRecognition:

    def __init__(self, beta: float = 1., astar: AStar = None, smoother: VelocitySmoother = None, cost: Cost = None):
        self._beta = beta
        self._astar = astar
        self._smoother = smoother
        self._cost = cost

    def update_goals_probabilities(self, goals_probabilities: GoalsProbabilities, trajectory: StateTrajectory, maneuver: Maneuver, scenario_map: Map) -> GoalsProbabilities :
        # not tested
        sum_likelihood = 0
        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            goal = goal_and_type[0]
            goal_type = goal_and_type[1]
            #4. and 5. Generate optimum trajectory from initial point and smooth it
            opt_trajectory = self.generate_trajectory(trajectory.initial_agent_state, goal, maneuver, scenario_map)
            #7. and 8. Generate optimum trajectory from last observed point and smooth it
            current_trajectory = self.generate_trajectory(trajectory.final_agent_state, goal, maneuver, scenario_map)
            #10. current_trajectory = join(trajectory, togoal_trajectory)
            current_trajectory.insert(trajectory)
            #6,9,10. likelihood = self.likelihood(current_trajectory, opt_trajectory)
            likelihood = self.likelihood(current_trajectory, opt_trajectory)
            prob *= likelihood
            sum_likelihood += likelihood

        # then divide prob by sum_likelihood
        for prob in goals_probabilities.goals_probabilities.values():
            prob = prob / sum_likelihood

        #raise NotImplementedError
        return goals_probabilities

    def generate_trajectory(self, state: AgentState, goal: Goal, maneuver: Maneuver, scenario_map: Map) -> VelocityTrajectory:
        # Q? where does the maneuver go in astar search algo?
        trajectory = self._astar.search(0, state, goal, scenario_map)
        self._smoother.load_trajectory(trajectory)
        trajectory.velocity = self._smoother.split_smooth()
        return trajectory

    def likelihood(self, current_trajectory : VelocityTrajectory, opt_trajectory: VelocityTrajectory, goal: Goal) -> float :
        # not tested
        r_current = self.reward(current_trajectory, goal)
        r_opt = self.reward(opt_trajectory, goal)
        math.exp(self._beta * (r_current - r_opt))

    def reward(self, trajectory: Trajectory, goal: Goal) -> float:
        return - self._cost.trajectory_cost(trajectory, goal)