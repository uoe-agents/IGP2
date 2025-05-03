import numpy as np
import logging
from typing import List, Dict, Tuple
from shapely.geometry import Point

from igp2.agents.traffic_agent import TrafficAgent
from igp2.core.trajectory import Trajectory, StateTrajectory
from igp2.core.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.core.goal import Goal, PointGoal, StoppingGoal, PointCollectionGoal
from igp2.core.vehicle import TrajectoryVehicle, Observation, Action
from igp2.core.util import Circle, find_lane_sequence
from igp2.core.cost import Cost
from igp2.core.velocitysmoother import VelocitySmoother
from igp2.planlibrary.maneuver import Maneuver
from igp2.planning.reward import Reward
from igp2.planning.mcts import MCTS
from igp2.recognition.astar import AStar
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.recognition.goalprobabilities import GoalsProbabilities

logger = logging.getLogger(__name__)


class MCTSAgent(TrafficAgent):

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 t_update: float,
                 scenario_map: Map,
                 goal: Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 kinematic: bool = False,
                 n_simulations: int = 5,
                 max_depth: int = 5,
                 store_results: str = 'final',
                 trajectory_agents: bool = True,
                 cost_factors: Dict[str, float] = None,
                 reward_factors: Dict[str, float] = None,
                 default_rewards: Dict[str, float] = None,
                 velocity_smoother: dict = None,
                 goal_recognition: dict = None,
                 stop_goals: bool = False):
        """ Create a new MCTS agent.

        Args:
            agent_id: THe ID of the agent to create
            initial_state: The initial state of the agent at the start of initialisation
            t_update: the time interval between runs of the planner
            scenario_map: The current road layout
            goal: The end goal of the agent
            view_radius: The radius of a circle in which the agent can see the other agents
            fps: The execution frequency of the environment
            kinematic: If True then use a kinematic vehicle, otherwise a trajectory vehicle.
            n_simulations: The number of simulations to perform in MCTS
            max_depth: The maximum search depth of MCTS (in macro actions)
            store_results: Whether to save the traces of the MCTS rollouts
            trajectory_agents: Whether to use trajectories or plans for non-egos in MCTS
            cost_factors: For trajectory cost calculations of ego in goal recognition
            reward_factors: Reward factors for MCTS rollouts
            velocity_smoother: Velocity smoother arguments. See: VelocitySmoother
            goal_recognition: Goal recognition parameters. See: GoalRecognition
            stop_goals: Whether to check for stopping goals.
        """
        super().__init__(agent_id, initial_state, goal, fps)
        if not kinematic:
            self._vehicle = TrajectoryVehicle(initial_state, self.metadata, fps)

        self._goal_probabilities = None
        self._observations = {}
        self._k = 0
        self._stop_goals = stop_goals

        self._view_radius = view_radius
        self._kmax = t_update * self._fps

        self._cost = Cost(factors=cost_factors) if cost_factors is not None else Cost()
        self._reward = Reward(factors=reward_factors, default_rewards=default_rewards) if reward_factors is not None \
            else Reward()

        self._astar = AStar(next_lane_offset=0.1)
        if velocity_smoother is None:
            velocity_smoother = {"vmin_m_s": 1, "vmax_m_s": 10, "n": 10, "amax_m_s2": 5, "lambda_acc": 10}
        self._smoother = VelocitySmoother(**velocity_smoother)

        if goal_recognition is None:
            goal_recognition = {"reward_as_difference": False, "n_trajectories": 2}
        self._goal_recognition = GoalRecognition(astar=self._astar,
                                                 smoother=self._smoother,
                                                 scenario_map=scenario_map,
                                                 cost=self._cost,
                                                 **goal_recognition)

        self._mcts = MCTS(scenario_map=scenario_map,
                          reward=self._reward,
                          n_simulations=n_simulations,
                          max_depth=max_depth,
                          store_results=store_results,
                          trajectory_agents=trajectory_agents)

        self._goals: List[Goal] = []

    def __repr__(self) -> str:
        return f"MCTSAgent(id={self.agent_id}, goal={self.goal})"

    def __str__(self) -> str:
        return repr(self)

    def done(self, observation: Observation):
        """ True if the agent has reached its goal. """
        return self.goal.reached(self.state.position)

    def reset(self):
        """ Reset the vehicle and macro action of the agent."""
        super(MCTSAgent, self).reset()
        self._vehicle = type(self.vehicle)(self._initial_state, self.metadata, self._fps)

    def update_plan(self, observation: Observation):
        """ Runs MCTS to generate a new sequence of macro actions to execute."""
        frame = observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        self._goal_probabilities = {aid: GoalsProbabilities(self._goals)
                                    for aid in frame.keys() if aid != self.agent_id}
        visible_region = Circle(frame[self.agent_id].position, self.view_radius)

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=self._goal_probabilities[agent_id],
                observed_trajectory=self._observations[agent_id][0],
                agent_id=agent_id,
                frame_ini=self._observations[agent_id][1],
                frame=frame,
                visible_region=visible_region)

            logger.info("")
            self._goal_probabilities[agent_id].log(logger)
            logger.info("")

        self._macro_actions, _ = self._mcts.search(
            agent_id=self.agent_id,
            goal=self.goal,
            frame=frame,
            meta=agents_metadata,
            predictions=self._goal_probabilities)
        self._current_macro_id = 0

    def next_action(self, observation: Observation) -> Action:
        """ Returns the next action for the agent.

        If the current macro actions has finished, then updates it.
        If no macro actions are left in the plan, or we have hit the planning time step, then calls goal recognition
        and MCTS. """
        self.update_observations(observation)

        if self._k >= self._kmax or self.current_macro is None or \
                (self.current_macro.done(observation) and self._current_macro_id >= len(self._macro_actions) - 1):
            self._goals = self.get_goals(observation)
            self.update_plan(observation)
            self.update_macro_action(
                self._macro_actions[0].macro_action_type,
                self._macro_actions[0].ma_args,
                observation)
            self._k = 0

        if self.current_macro.done(observation):
            self._advance_macro(observation)

        self._k += 1
        return self.current_macro.next_action(observation)

    def update_observations(self, observation: Observation):
        frame = observation.frame
        for aid, agent_state in frame.items():
            try:
                # TODO check that this will handle agents disappearing and reappearing from field of view successfully

                # We remove the agent after a single frame of it not being observed, to handle despawned
                # agents and agents that leave the field of view.
                # As a future improvement we could have the agents that leave the observation radius stay in memory
                # for a given amount of time. We could do this by comparing the observation time with the last time
                # each agent was observed. We should also use the alive/dead attribute for despawned agents.
                self._observations[aid][0].add_state(agent_state)
            except KeyError:
                self._observations[aid] = (StateTrajectory(fps=self._fps, states=[agent_state]), frame.copy())

        for aid in list(self._observations.keys()):
            if aid not in frame:
                self._observations.pop(aid)

    def get_goals(self,
                  observation: Observation,
                  threshold: float = 2.0) -> List[Goal]:
        """Retrieve all possible goals reachable from the current position on the map in any direction. If more than
        one goal is found on a single lane, then only choose the one furthest along the midline of the lane.

        Args:
            observation: Observation of the environment
            threshold: The goal checking threshold
        """
        scenario_map = observation.scenario_map
        frame = observation.frame
        state = frame[self.agent_id]
        view_circle = Point(*state.position).buffer(self.view_radius)

        possible_goals = []

        # Retrieve relevant roads and check intersection of its lanes' midlines
        for road in scenario_map.roads.values():
            if not road.boundary.intersects(view_circle):
                continue

            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == 0 or lane.type != "driving":
                        continue

                    new_point = None

                    # First check if the lane intersects the view boundary anywhere
                    intersection = lane.midline.intersection(view_circle.boundary)
                    if not intersection.is_empty:
                        if hasattr(intersection, "geoms"):
                            max_distance = np.inf
                            for point in intersection.geoms:
                                if lane.distance_at(point) < max_distance:
                                    new_point = point
                        else:
                            new_point = intersection

                    # If not, and the lane is completely within the circle then choose the lane end-point
                    elif view_circle.contains(lane.boundary):
                        # if lane.link.predecessor is None:
                        #     new_point = lane.midline.coords[-1]
                        if lane.link.successor is None:
                            new_point = Point(lane.midline.coords[-1])
                        else:
                            continue

                    else:
                        continue

                    # Do not add point if within threshold distance to an existing goal
                    new_point = np.array(new_point.coords[0])
                    if not any([np.allclose(new_point, g.center, atol=threshold) for _, g in possible_goals]):
                        new_goal = PointGoal(new_point, threshold=threshold)
                        possible_goals.append((lane, new_goal))

        # Add stopping goals
        stopping_goals = []
        if self._stop_goals:
            for aid, s in frame.items():
                if aid == self.agent_id:
                    continue

                # First, for all agents that are stopped
                if s.speed < Trajectory.VELOCITY_STOP:
                    stopping_goals.append(StoppingGoal(s.position, threshold=threshold))

                # If there is a stopped vehicle ahead and there is a path to the goal (had the stopped
                #  vehicle not been there), then add a stopping goal behind the stopped vehicle.
                current_lane = observation.scenario_map.best_lane_at(s.position, s.heading)
                for lane, goal in possible_goals:
                    lanes_to_goal = find_lane_sequence(current_lane, lane, goal)
                    if not lanes_to_goal:
                        continue
                    vehicle_in_front, distance, lane_ls = Maneuver.get_vehicle_in_front(aid, frame, lanes_to_goal)
                    if vehicle_in_front is not None and frame[vehicle_in_front].speed - Trajectory.VELOCITY_STOP < 0.05:
                        ds = lane_ls.project(Point(frame[vehicle_in_front].position))
                        backtrack_length = frame[vehicle_in_front].metadata.length / 2 + 3 + self.metadata.length / 2
                        backtrack_ds = max(self.metadata.length / 2 + 1e-3, ds - backtrack_length)
                        stopping_point = np.array(lane_ls.interpolate(backtrack_ds).coords[0])
                        if not any([np.allclose(stopping_point, g.center, atol=threshold) for g in stopping_goals]):
                            new_goal = StoppingGoal(stopping_point, threshold=threshold)
                            stopping_goals.append(new_goal)

        # Group goals that are in neighbouring lanes
        goals = []
        used = []
        for lane, goal in possible_goals:
            if goal in used:
                continue

            neighbouring_goals = [goal]
            for other_lane, other_goal in possible_goals:
                if goal == other_goal:
                    continue

                if lane.parent_road == other_lane.parent_road and np.abs(lane.id - other_lane.id) == 1:
                    # TODO: This could group goals that are in neighbouring lanes but very far apart still.
                    neighbouring_goals.append(other_goal)
                    used.append(other_goal)

            if len(neighbouring_goals) > 1:
                goals.append(PointCollectionGoal(neighbouring_goals))
            else:
                goals.append(goal)

        return goals + stopping_goals

    def _advance_macro(self, observation: Observation):

        if not self._macro_actions:
            raise RuntimeError("Agent has no macro actions.")

        self._current_macro_id += 1
        if self._current_macro_id >= len(self._macro_actions):
            raise RuntimeError("No more macro actions to execute.")
        else:
            next_macro = self._macro_actions[self._current_macro_id]
            self.update_macro_action(next_macro.macro_action_type, next_macro.ma_args, observation)

    @property
    def view_radius(self) -> float:
        """ The view radius of the agent. """
        return self._view_radius

    @property
    def observations(self) -> Dict[int, Tuple[StateTrajectory, AgentState]]:
        """Returns the ego's knowledge about other agents, sorted in a dictionary with keys
        corresponding to agents ids. It stores the trajectory observed so far and the frame
        at which each agent was initially observed. Currently, any agent out of view is immediately forgotten."""
        return self._observations

    @property
    def possible_goals(self) -> List[Goal]:
        """ Return the current list of possible goals. """
        return self._goals

    @property
    def goal_probabilities(self) -> Dict[int, GoalsProbabilities]:
        """ Return the currently stored goal prediction probabilities of the ego."""
        return self._goal_probabilities

    @property
    def mcts(self) -> "MCTS":
        """ Return the MCTS planner of the agent. """
        return self._mcts

    @property
    def reward(self) -> Reward:
        """ Return the reward function of the agent. """
        return self._reward
