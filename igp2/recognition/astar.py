import traceback

import numpy as np
import heapq
import logging
from typing import Callable, List, Dict, Tuple

from shapely.geometry import Point, LineString

from igp2.agentstate import AgentState
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.planlibrary.maneuver import Maneuver
from igp2.trajectory import VelocityTrajectory

logger = logging.getLogger(__name__)


class AStar:
    """ Class implementing A* search over trajectories to goals. """

    def __init__(self,
                 cost_function: Callable[[VelocityTrajectory, PointGoal], float] = None,
                 heuristic_function: Callable[[VelocityTrajectory, PointGoal], float] = None,
                 next_lane_offset: float = 0.15,
                 max_iter: int = 100):
        """ Initialises a new A* search class with the given parameters. The search frontier is ordered according to the
        formula f = g + h.

        Args:
            next_lane_offset: A small offset used to reach the next lane when search to the end of a lane
            cost_function: The cost function g
            heuristic_function: The heuristic function h
            max_iter: The maximum number of iterations A* is allowed to run
        """
        self.next_lane_offset = next_lane_offset
        self.max_iter = max_iter

        self._g = AStar.trajectory_duration if cost_function is None else cost_function
        self._h = AStar.time_to_goal if heuristic_function is None else heuristic_function
        self._f = self.cost_function

    def search(self, agent_id: int, frame: Dict[int, AgentState], goal: PointGoal, scenario_map: Map,
               n_trajectories: int = 1,
               current_maneuver: Maneuver = None) -> Tuple[List[VelocityTrajectory], List[List[MacroAction]]]:
        """ Run A* search from the current frame to find trajectories to the given goal.

        Args:
            agent_id: The agent to plan for
            frame: State of the environment to search from
            goal: The target goal
            scenario_map: The Map of the scenario
            n_trajectories: The number of trajectories to return
            current_maneuver: The currently executed maneuver of the agent

        Returns:
            List of VelocityTrajectories ordered in increasing order of cost. The best trajectory is at index 0, while
            the worst is at index -1
        """
        solutions = []
        if current_maneuver is not None:
            frame = Maneuver.play_forward_maneuver(agent_id, scenario_map, frame, current_maneuver)

        frontier = [(0.0, ([], frame))]
        iterations = 0
        while frontier and len(solutions) < n_trajectories and iterations < self.max_iter:
            iterations += 1
            cost, (actions, frame) = heapq.heappop(frontier)

            # Check termination condition
            if goal.reached(Point(frame[agent_id].position)):
                if not actions:
                    logger.debug(f"AID {agent_id} at {goal} already.")
                else:
                    logger.debug(f"Solution found for AID {agent_id} to {goal}: {actions}")
                    solutions.append(actions)
                continue

            # Check if current position is valid
            if not scenario_map.roads_at(frame[agent_id].position):
                continue

            # Check if path has self-intersection and in a roundabout
            if actions and scenario_map.in_roundabout(frame[agent_id].position, frame[agent_id].heading):
                trajectory = self._full_trajectory(actions)
                if not LineString(trajectory.path).is_simple: continue

            for macro_action in MacroAction.get_applicable_actions(frame[agent_id], scenario_map):
                for ma_args in macro_action.get_possible_args(frame[agent_id], scenario_map, goal.center):
                    try:
                        new_ma = macro_action(agent_id=agent_id, frame=frame, scenario_map=scenario_map,
                                              open_loop=True, **ma_args)

                        new_actions = actions + [new_ma]
                        new_trajectory = self._full_trajectory(new_actions)
                        new_frame = MacroAction.play_forward_macro_action(agent_id, scenario_map, frame, new_ma)
                        new_frame[agent_id] = new_trajectory.final_agent_state
                        new_cost = self._f(new_trajectory, goal)

                        heapq.heappush(frontier, (new_cost, (new_actions, new_frame)))
                    except Exception as e:
                        logger.debug(str(e))
                        logger.debug(traceback.format_exc())
                        continue

        trajectories = [self._full_trajectory(mas) for mas in solutions]
        return trajectories, solutions

    def cost_function(self, trajectory: VelocityTrajectory, goal: PointGoal) -> float:
        return self._g(trajectory, goal) + self._h(trajectory, goal)

    @staticmethod
    def trajectory_duration(trajectory: VelocityTrajectory, goal: PointGoal) -> float:
        return trajectory.duration

    @staticmethod
    def time_to_goal(trajectory: VelocityTrajectory, goal: PointGoal) -> float:
        return np.linalg.norm(trajectory.path[-1] - goal.center) / Maneuver.MAX_SPEED

    def _add_offset_point(self, trajectory):
        """ Add a small step at the end of the trajectory to reach within the boundary of the next lane. """
        heading = trajectory.heading[-1]
        direction = np.array([np.cos(heading), np.sin(heading)])
        point = trajectory.path[-1] + self.next_lane_offset * direction
        velocity = trajectory.velocity[-1]
        trajectory.extend((np.array([point]), np.array([velocity])))

    def _full_trajectory(self, macro_actions: List[MacroAction]):
        path = np.empty((0, 2), float)
        velocity = np.empty((0,), float)

        # Join trajectories of macro actions
        for ma in macro_actions:
            trajectory = ma.get_trajectory()
            path = np.concatenate([path[:-1], trajectory.path], axis=0)
            velocity = np.concatenate([velocity[:-1], trajectory.velocity])

        # Add final offset point
        full_trajectory = VelocityTrajectory(path, velocity)
        self._add_offset_point(full_trajectory)

        return full_trajectory
