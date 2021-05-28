import numpy as np
import math
from typing import Callable, List, Dict, Tuple

from igp2.agent import AgentState, TrajectoryAgent
from igp2.cost import Cost
from igp2.goal import Goal, PointGoal
from igp2.trajectory import Trajectory, VelocityTrajectory
from igp2.planlibrary.maneuver import Maneuver

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

    def __init__(self, beta: float = 1.):
        self._beta = beta
        raise NotImplementedError

    def update_goals_probabilities(self, goals_probabilities: GoalsProbabilities, trajectory: TrajectoryAgent, maneuver: Maneuver) -> GoalsProbabilities :
        
        sum_likelihood = 0
        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            
            #4. opt_trajectory = Astar_search(Maneuver, trajectory[0])
            #5. opt_trajectory = VelocitySmoother(opt_trajectory)
            #7. togoal_trajectory = Astar_search(Maneuver, trajectory[-1])
            #8. togoal_trajectory = VelocitySmoother(togoal_trajectory)
            #10. current_trajectory = join(trajectory, togoal_trajectory)
            #6,9,10. likelihood = self.likelihood(current_trajectory, opt_trajectory)
            # prob *= likelihood
            # sum_likelihood += likelihood
            print("boo")

        # then divide prob by sum_likelihood    

        raise NotImplementedError

    def likelihood(self, current_trajectory : VelocityTrajectory, opt_trajectory: VelocityTrajectory, goal: Goal) -> float :
        # not tested
        r_current = self.reward(current_trajectory, goal)
        r_opt = self.reward(opt_trajectory, goal)
        math.exp(self._beta * (r_current - r_opt))

    def reward(self, trajectory: Trajectory, goal: Goal) -> float:
        # TODO add possibility to parametrise cost coefficients here
        # not tested
        cost = Cost()
        return - cost.trajectory_cost(trajectory, goal)

from igp2.goal import PointGoal
from shapely.geometry import Point

goals_data = [[17.40, -4.97],
            [75.18, -56.65],
            [62.47, -17.54]]

goal_types = [["turn-right", "straight-on"],
                 ["turn-left", "straight-on", "u-turn"],
                 ["turn-left", "turn-right"]]

goals = []
for goal_data in goals_data:
    point = Point(np.array(goal_data))
    goals.append(PointGoal(point, 1.))

#print(goals)

goals_prob = GoalsProbabilities(goals, goal_types)

#print(goals_prob._goals_and_types)
print(goals_prob._goal_probabilities)

