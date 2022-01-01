import numpy as np

from typing import List, Dict, Tuple, Iterable
from shapely.geometry import Point

from shapely.geometry import Point

from igp2.agents.agentstate import AgentMetadata, AgentState
from igp2.agents.macro_agent import MacroAgent
from igp2.cost import Cost
from igp2.goal import Goal, PointGoal
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.planning.mcts import MCTS
from igp2.recognition.astar import AStar, Circle
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.trajectory import StateTrajectory
from igp2.vehicle import Observation, Action, TrajectoryVehicle
from igp2.velocitysmoother import VelocitySmoother


class MCTSAgent(MacroAgent):

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 t_update: float,
                 scenario_map: Map,
                 goal: Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 cost_factors: Dict[str, float] = None,
                 n_simulations: int = 5,
                 max_depth: int = 3,
                 store_results: str = 'final',
                 goals: Dict[int, Goal] = None):
        """ Create a new MCTS agent.

        Args:
            agent_id: THe ID of the agent to create
            initial_state: The initial state of the agent at the start of initialisation
            t_update: the time interval between runs of the planner
            scenario_map: The current road layout
            goal: The end goal of the agent
            view_radius: The radius of a circle in which the agent can see the other agents
            fps: The execution frequency of the environment
            cost_factors: For trajectory cost calculations of ego in MCTS
            n_simulations: The number of simulations to perform in MCTS
            max_depth: The maximum search depth of MCTS (in macro actions)
            store_results: Whether to save the traces of the MCTS rollouts
        """
        super().__init__(agent_id, initial_state, goal, fps)
        self._vehicle = TrajectoryVehicle(initial_state, self.metadata, fps)
        self._current_macro_id = 0
        self._macro_actions = None
        self._goal_probabilities = None
        self._observations = {}
        self._k = 0
        self._view_radius = view_radius
        self._kmax = t_update * self._fps
        self._cost = Cost(factors=cost_factors) if cost_factors is not None else Cost()
        self._astar = AStar(next_lane_offset=0.25)
        self._smoother = VelocitySmoother(vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10)
        self._goal_recognition = GoalRecognition(astar=self._astar, smoother=self._smoother, scenario_map=scenario_map,
                                                 cost=self._cost, reward_as_difference=True, n_trajectories=2)
        self._mcts = MCTS(scenario_map, n_simulations=n_simulations, max_depth=max_depth, store_results=store_results)

        self._goals = goals

    def done(self, observation: Observation):
        return self.goal.reached(Point(self.state.position))

    def update_plan(self, observation: Observation):
        """ Runs MCTS to generate a new sequence of macro actions to execute."""
        frame = observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        self._goal_probabilities = {aid: GoalsProbabilities(self._goals) for aid in frame.keys()}
        visible_region = Circle(frame[self.agent_id].position, self.view_radius)
        for agent_id in frame:
            state = frame[agent_id]
            agent_distance = np.linalg.norm(state.position - frame[self.agent_id].position)
            if agent_id == self.agent_id or agent_distance >= self.view_radius:
                continue

            self._goal_recognition.update_goals_probabilities(self._goal_probabilities[agent_id],
                                                              self._observations[agent_id][0],
                                                              agent_id, self._observations[agent_id][1], frame, None,
                                                              visible_region=visible_region)
        # TODO: Use sub-goal instead generated along the path to the final goal
        self._macro_actions = self._mcts.search(self.agent_id, self.goal, frame,
                                                agents_metadata, self._goal_probabilities)
        self._current_macro_id = 0

    def next_action(self, observation: Observation) -> Action:

        self.update_observations(observation)

        if self._k >= self._kmax or self.current_macro is None or \
                (self.current_macro.done(observation) and self._current_macro_id == len(self._macro_actions) - 1):
            self._goals = self.get_goals(observation)
            self.update_plan(observation)
            self.update_macro_action(self._macro_actions[0], observation)
            self._k = 0

        if self.current_macro.done(observation): self._advance_macro(observation)

        self._k += 1
        self.trajectory_cl.add_state(observation.frame[self.agent_id])
        self._vehicle.execute_action(next_state=observation.frame[self.agent_id])
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
                self._observations[aid] = (StateTrajectory(fps=self._fps, frames=[agent_state]), frame)

        for aid in list(self._observations.keys()):
            if aid not in frame: self._observations.pop(aid)

    def get_goals(self, observation: Observation) -> List[Goal]:
        """Retrieve all possible goals reachable from the current position on the map in any direction. If more than
        one goal is found on a single lane, then only choose the one furthest along the midline of the lane.

        Args:
            observation: Observation of the environment
        """
        # TODO Replace with box goals that cover all lanes in same direction

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
                    possible_goals.append(PointGoal(np.array(new_point), threshold=2.0))

        return possible_goals

    def _advance_macro(self, observation: Observation):

        if not self._macro_actions:
            raise RuntimeError("Agent has no macro actions.")

        self._current_macro_id += 1
        if self._current_macro_id >= len(self._macro_actions):
            raise RuntimeError("No more macro actions to execute.")
        else:
            self.update_macro_action(self._macro_actions[self._current_macro_id], observation)

    @property
    def view_radius(self) -> float:
        """ The view radius of the agent. """
        return self._view_radius

    @property
    def macro_actions(self) -> List[MacroAction]:
        """ The current macro actions to be executed by the agent. """
        return self._macro_actions

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
