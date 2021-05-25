import numpy as np
import heapq
from typing import Callable, List, Dict

from igp2.agent import AgentState
from igp2.cost import Cost
from igp2.goal import Goal, PointGoal
from igp2.trajectory import VelocityTrajectory


class AStar:
    """ Class implementing A* search over trajectories to goals. """
    def __init__(self,
                 n_trajectories: int = 2,
                 cost_function: Callable[[VelocityTrajectory, PointGoal], float] = None,
                 heuristic_function: Callable[[VelocityTrajectory, PointGoal], float] = None,
                 trajectory_cost: Cost = None,
                 next_lane_offset: float = 0.1,
                 goal_threshold: float = 0.1):
        """ Initialises a new A* search class with the given parameters. The search frontier is ordered according to the
        formula f = g + h.

        Args:
            n_trajectories: The number of trajectories to return
            next_lane_offset: A small offset used to reach the next lane when search to the end of a lane
            goal_threshold: Threshold to check goal completion
            cost_function: The cost function g
            heuristic_function: The heuristic function h
            trajectory_cost: The exact cost signal of the generated trajectory
        """
        self.n_trajectories = n_trajectories
        self.next_lane_offset = next_lane_offset
        self.goal_threshold = goal_threshold
        self.cost = Cost() if trajectory_cost is None else trajectory_cost

        self._g = self.trajectory_duration if cost_function is None else cost_function
        self._h = self.time_to_goal if heuristic_function is None else heuristic_function
        self._f = self.cost_function

        self._frontier = []

    def search(self, frame: Dict[int, AgentState], goal: PointGoal) -> List[VelocityTrajectory]:
        """ Run A* search from the current frame to find trajectories to the given goal.

        Args:
            frame: State of the environment to search from
            goal: The target goal

        Returns:
            List of VelocityTrajectories ordered in increasing order of cost. The best trajectory is at index 0, while
            the worst is at index -1
        """
        raise NotImplementedError

    def cost_function(self, trajectory: VelocityTrajectory, goal: PointGoal) -> float:
        return self._g(trajectory, goal) + self._h(trajectory, goal)

    def trajectory_duration(self, trajectory: VelocityTrajectory, goal: PointGoal) -> float:
        return trajectory.duration

    def time_to_goal(self, trajectory: VelocityTrajectory, goal: PointGoal) -> float:
        return np.linalg.norm(trajectory.path[-1] - goal.center)

