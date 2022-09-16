import igp2 as ip
import numpy as np
from typing import List, Dict, Tuple, Iterable
from shapely.geometry import Point

from igp2.agents.traffic_agent import TrafficAgent


class MCTSAgent(TrafficAgent):

    def __init__(self,
                 agent_id: int,
                 initial_state: ip.AgentState,
                 t_update: float,
                 scenario_map: ip.Map,
                 goal: ip.Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 cost_factors: Dict[str, float] = None,
                 reward_factors: Dict[str, float] = None,
                 n_simulations: int = 5,
                 max_depth: int = 3,
                 store_results: str = 'final'):
        """ Create a new MCTS agent.

        Args:
            agent_id: THe ID of the agent to create
            initial_state: The initial state of the agent at the start of initialisation
            t_update: the time interval between runs of the planner
            scenario_map: The current road layout
            goal: The end goal of the agent
            view_radius: The radius of a circle in which the agent can see the other agents
            fps: The execution frequency of the environment
            cost_factors: For trajectory cost calculations of ego in goal recognition
            reward_factors: Reward factors for MCTS rollouts
            n_simulations: The number of simulations to perform in MCTS
            max_depth: The maximum search depth of MCTS (in macro actions)
            store_results: Whether to save the traces of the MCTS rollouts
        """
        super().__init__(agent_id, initial_state, goal, fps)
        self._vehicle = ip.TrajectoryVehicle(initial_state, self.metadata, fps)
        self._current_macro_id = 0
        self._macro_actions = None
        self._goal_probabilities = None
        self._observations = {}
        self._k = 0
        self._view_radius = view_radius
        self._kmax = t_update * self._fps
        self._cost = ip.Cost(factors=cost_factors) if cost_factors is not None else ip.Cost()
        self._reward = ip.Reward(factors=reward_factors) if reward_factors is not None else ip.Reward()
        self._astar = ip.AStar(next_lane_offset=0.25)
        self._smoother = ip.VelocitySmoother(vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10)
        self._goal_recognition = ip.GoalRecognition(astar=self._astar, smoother=self._smoother,
                                                    scenario_map=scenario_map,
                                                    cost=self._cost, reward_as_difference=False, n_trajectories=2)
        self._mcts = ip.MCTS(scenario_map, n_simulations=n_simulations, max_depth=max_depth,
                             store_results=store_results, reward=self._reward)

        self._goals: List[ip.Goal] = []

    def done(self, observation: ip.Observation):
        """ True if the agent has reached its goal. """
        return self.goal.reached(self.state.position)

    def update_plan(self, observation: ip.Observation):
        """ Runs MCTS to generate a new sequence of macro actions to execute."""
        frame = observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        self._goal_probabilities = {aid: ip.GoalsProbabilities(self._goals)
                                    for aid in frame.keys() if aid != self.agent_id}
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            self._goal_recognition.update_goals_probabilities(self._goal_probabilities[agent_id],
                                                              self._observations[agent_id][0],
                                                              agent_id, self._observations[agent_id][1], frame, None,
                                                              visible_region=visible_region)
        self._macro_actions = self._mcts.search(self.agent_id, self.goal, frame,
                                                agents_metadata, self._goal_probabilities)
        self._current_macro_id = 0

    def next_action(self, observation: ip.Observation) -> ip.Action:
        """ Returns the next action for the agent.

        If the current macro actions has finished, then updates it.
        If no macro actions are left in the plan or we have hit the planning time step, then calls goal recognition
        and MCTS. """
        self.update_observations(observation)

        if self._k >= self._kmax or self.current_macro is None or \
                (self.current_macro.done(observation) and self._current_macro_id == len(self._macro_actions) - 1):
            self.get_goals(observation)
            self.update_plan(observation)
            self.update_macro_action(self._macro_actions[0].macro_action_type,
                                     self._macro_actions[0].ma_args,
                                     observation)
            self._k = 0

        if self.current_macro.done(observation):
            self._advance_macro(observation)

        self._k += 1
        self.trajectory_cl.add_state(observation.frame[self.agent_id])
        self._vehicle.execute_action(next_state=observation.frame[self.agent_id])
        return self.current_macro.next_action(observation)

    def update_observations(self, observation: ip.Observation):
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
                self._observations[aid] = (ip.StateTrajectory(fps=self._fps, frames=[agent_state]), frame)

        for aid in list(self._observations.keys()):
            if aid not in frame: self._observations.pop(aid)

    def get_goals(self,
                  observation: ip.Observation,
                  threshold: float = 2.0) -> List[ip.Goal]:
        """Retrieve all possible goals reachable from the current position on the map in any direction. If more than
        one goal is found on a single lane, then only choose the one furthest along the midline of the lane.

        Args:
            observation: Observation of the environment
            threshold: The goal checking threshold
        """
        scenario_map = observation.scenario_map
        state = observation.frame[self.agent_id]
        view_circle = Point(*state.position).buffer(self.view_radius)

        # Retrieve relevant roads and check intersection of its lanes' midlines
        possible_goals = []
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
                        if isinstance(intersection, Iterable):
                            max_distance = np.inf
                            for point in intersection:
                                if lane.distance_at(point) < max_distance:
                                    new_point = point
                        else:
                            new_point = intersection

                    # If not, and the lane is completely within the circle then choose the lane end-point
                    elif view_circle.contains(lane.boundary):
                        # if lane.link.predecessor is None:
                        #     new_point = lane.midline.coords[-1]
                        if lane.link.successor is None:
                            new_point = lane.midline.coords[-1]
                        else:
                            continue

                    else:
                        continue

                    # Do not add point if within threshold distance to an existing goal
                    if not any([np.allclose(new_point, g.center, atol=threshold) for _, g in possible_goals]):
                        new_goal = ip.PointGoal(np.array(new_point), threshold=threshold)
                        possible_goals.append((lane, new_goal))

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
                    neighbouring_goals.append(other_goal)
                    used.append(other_goal)

            if len(neighbouring_goals) > 1:
                goals.append(ip.PointCollectionGoal(neighbouring_goals))
            else:
                goals.append(goal)

        self._goals = goals
        return goals

    def _advance_macro(self, observation: ip.Observation):

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
    def observations(self) -> Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]]:
        """Returns the ego's knowledge about other agents, sorted in a dictionary with keys
        corresponding to agents ids. It stores the trajectory observed so far and the frame
        at which each agent was initially observed. Currently, any agent out of view is immediately forgotten."""
        return self._observations

    @property
    def possible_goals(self) -> List[ip.Goal]:
        """ Return the current list of possible goals. """
        return self._goals

    @property
    def goal_probabilities(self) -> Dict[int, ip.GoalsProbabilities]:
        """ Return the currently stored goal prediction probabilities of the ego."""
        return self._goal_probabilities
