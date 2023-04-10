import traceback
import igp2 as ip
import numpy as np
import heapq
import logging
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Tuple

from shapely.geometry import LineString, Point
from scipy.spatial import distance_matrix

logger = logging.getLogger(__name__)


class AStar:
    """ Class implementing A* search over trajectories to goals. """

    def __init__(self,
                 cost_function: Callable[[ip.VelocityTrajectory, ip.PointGoal], float] = None,
                 heuristic_function: Callable[[ip.VelocityTrajectory, ip.PointGoal], float] = None,
                 next_lane_offset: float = 0.01,
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

    def search(self,
               agent_id: int,
               frame: Dict[int, ip.AgentState],
               goal: ip.Goal,
               scenario_map: ip.Map,
               n_trajectories: int = 1,
               open_loop: bool = True,
               debug: bool = False,
               visible_region: ip.Circle = None) -> Tuple[List[ip.VelocityTrajectory], List[List[ip.MacroAction]]]:
        """ Run A* search from the current frame to find trajectories to the given goal.

        Args:
            agent_id: The agent to plan for
            frame: State of the environment to search from
            goal: The target goal
            scenario_map: The Map of the scenario
            n_trajectories: The number of trajectories to return
            open_loop: Whether to generate open loop or closed loop macro actions in the end
            debug: If True, then plot the evolution of the frontier at each step
            visible_region: Region of the map that is visible to the ego vehicle

        Returns:
            List of VelocityTrajectories ordered in increasing order of cost. The best trajectory is at index 0, while
            the worst is at index -1
        """
        solutions = []
        frontier = [(0.0, ([], frame))]
        iterations = 0
        while frontier and len(solutions) < n_trajectories and iterations < self.max_iter:
            iterations += 1
            cost, (actions, frame) = heapq.heappop(frontier)

            # Check termination condition
            trajectory = self._full_trajectory(actions, add_offset_point=False)
            if self.goal_reached(goal, trajectory) and \
                    (not isinstance(goal, ip.StoppingGoal) or
                     trajectory.duration >= ip.StopMA.DEFAULT_STOP_DURATION):
                if not actions:
                    logger.info(f"AID {agent_id} at {goal} already.")
                else:
                    logger.info(f"Solution found for AID {agent_id} to {goal}: {actions}")
                    solutions.append(actions)
                continue

            # Check if current position is valid
            if not scenario_map.roads_at(frame[agent_id].position):
                continue

            # Check if path has self-intersection
            if actions:  # and scenario_map.in_roundabout(frame[agent_id].position, frame[agent_id].heading):
                if self._check_looping(trajectory, actions[-1]):
                    continue

                if debug:
                    ip.plot_map(scenario_map, midline=True)
                    for aid, a in frame.items():
                        plt.plot(*a.position, marker="o")
                        plt.text(*a.position, aid)
                    plt.scatter(trajectory.path[:, 0], trajectory.path[:, 1],
                                c=trajectory.velocity, cmap=plt.cm.get_cmap('Reds'), vmin=-4, vmax=20, s=8)
                    plt.plot(goal.center.x, goal.center.y, marker="x")
                    plt.title(f"agent {agent_id} -> {goal}: {actions}")
                    plt.show()

            for macro_action in ip.MacroAction.get_applicable_actions(frame[agent_id], scenario_map, goal):
                for ma_args in macro_action.get_possible_args(frame[agent_id], scenario_map, goal):
                    try:
                        ma_args["open_loop"] = open_loop
                        config = ip.MacroActionConfig(ma_args)
                        new_ma = macro_action(config, agent_id=agent_id, frame=frame, scenario_map=scenario_map)

                        new_actions = actions + [new_ma]
                        new_trajectory = self._full_trajectory(new_actions)

                        # Check if has passed through region and went outside region already
                        if not self._check_in_region(new_trajectory, visible_region):
                            continue

                        new_frame = ip.MacroAction.play_forward_macro_action(agent_id, scenario_map, frame, new_ma)
                        new_frame[agent_id] = new_trajectory.final_agent_state
                        new_cost = self._f(new_trajectory, goal)

                        heapq.heappush(frontier, (new_cost, (new_actions, new_frame)))
                    except Exception as e:
                        logger.debug(str(e))
                        logger.debug(traceback.format_exc())
                        continue

        trajectories = [self._full_trajectory(mas, add_offset_point=False) for mas in solutions]
        return trajectories, solutions

    def cost_function(self, trajectory: ip.VelocityTrajectory, goal: ip.Goal) -> float:
        return self._g(trajectory, goal) + self._h(trajectory, goal)

    @staticmethod
    def trajectory_duration(trajectory: ip.VelocityTrajectory, goal: ip.Goal) -> float:
        return trajectory.duration

    @staticmethod
    def time_to_goal(trajectory: ip.VelocityTrajectory, goal: ip.Goal) -> float:
        return goal.distance(Point(trajectory.path[-1])) / ip.Maneuver.MAX_SPEED

    @staticmethod
    def goal_reached(goal: ip.Goal, trajectory: ip.VelocityTrajectory) -> bool:
        if trajectory is None:
            return False

        if goal.reached(trajectory.path[-1]):
            return True
        else:
            return goal.passed_through_goal(trajectory)

    def _full_trajectory(self, macro_actions: List[ip.MacroAction], add_offset_point: bool = True):
        if not macro_actions:
            return None

        path = np.empty((0, 2), float)
        velocity = np.empty((0,), float)

        # Join trajectories of macro actions
        for ma in macro_actions:
            trajectory = ma.get_trajectory()
            path = np.concatenate([path[:-1], trajectory.path], axis=0)
            velocity = np.concatenate([velocity[:-1], trajectory.velocity])

        # Add final offset point
        full_trajectory = ip.VelocityTrajectory(path, velocity)
        if add_offset_point:
            ip.util.add_offset_point(full_trajectory, self.next_lane_offset)

        return full_trajectory

    def _check_looping(self, trajectory: ip.VelocityTrajectory, final_action: ip.MacroAction) -> bool:
        """ Checks whether the final action brought us back to somewhere we had already visited. """
        final_path = final_action.get_trajectory().path[::-1]
        previous_path = trajectory.path[:-len(final_path)]
        ds = distance_matrix(previous_path, final_path)
        close_points = np.sum(np.isclose(ds, 0.0, atol=2 * ip.Maneuver.POINT_SPACING), axis=1)
        return np.count_nonzero(close_points) > 2 / ip.Maneuver.POINT_SPACING

    def _check_in_region(self, trajectory: ip.VelocityTrajectory, visible_region: ip.Circle) -> bool:
        """ Checks whether the trajectory is in the visible region. Ignores the initial section outside of the visible
        region, as this often happens when the vehicle calculates the optimal trajectory from the first observed point
        of a vehicle."""
        if visible_region is None:
            return True

        dists = np.linalg.norm(trajectory.path[:-1] - visible_region.centre, axis=1)  # remove ending off offset point
        in_region = dists <= visible_region.radius + 1  # Add 1m for error
        if True in in_region:
            first_in_idx = np.nonzero(in_region)[0][0]
            in_region = in_region[first_in_idx:]
            return np.all(in_region)
        return True
