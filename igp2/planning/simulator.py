from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.agent import TrajectoryAgent, Agent, MacroAgent
from igp2.agentstate import AgentState, AgentMetadata
from igp2.goal import Goal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.macro_action import MacroAction
from igp2.trajectory import Trajectory, StateTrajectory, VelocityTrajectory
from igp2.vehicle import Observation
from gui.tracks_import import calculate_rotated_bboxes


class Simulator:
    """ Lightweight environment simulator useful for rolling out scenarios in MCTS.

    One agent is designated as the ego vehicle, while the other agents follow predefined trajectories calculated
    during goal recognition. Simulation is performed at a given frequency with collision checking.
    """

    def __init__(self,
                 ego_id: int,
                 initial_frame: Dict[int, AgentState],
                 metadata: Dict[int, AgentMetadata],
                 scenario_map: Map,
                 fps: int = 10,
                 open_loop_agents: bool = False):
        """Initialise new light-weight simulator with the given params.

        Args:
            ego_id: ID of the ego vehicle
            initial_frame: initial state of the environment
            metadata: metadata describing the agents in the environment
            scenario_map: current road layout
            fps: frame rate of simulation
            open_loop_agents: Whether non-ego agents follow open-loop control
        """
        assert ego_id in initial_frame, f"Ego ID {ego_id} is not in the initial frame!"
        assert ego_id in metadata, f"Ego ID {ego_id} not among given metadata!"

        self._scenario_map = scenario_map
        self._ego_id = ego_id
        self._initial_frame = initial_frame
        self._metadata = metadata
        self._fps = fps
        self._open_loop = open_loop_agents
        self._agents = self._create_agents()

    def update_trajectory(self, agent_id: int, new_trajectory: Trajectory):
        """ Update the predicted trajectory of the non-ego agent. Has no effect for ego or if agent_id not in agents

        Args:
            agent_id: ID of agent to update
            new_trajectory: new trajectory for agent
        """
        if agent_id in self._agents and agent_id != self._ego_id:
            self._agents[agent_id].set_trajectory(new_trajectory)

    def update_ego_action(self, action: MacroAction, frame: Dict[int, AgentState]):
        """ Update the current macro action of the ego vehicle.

        Args:
            action: new macro action to execute
            frame: Current state of the environment
        """
        observation = Observation(frame, self._scenario_map)
        self._agents[self._ego_id].update_macro_action(action, observation)

    def update_ego_goal(self, goal: Goal):
        """ Update the final goal of the ego vehicle.

        Args:
            goal: new goal to reach
        """
        self._agents[self._ego_id].update_goal(goal)

    def reset(self):
        """ Reset the internal states of the environment to initialisation defaults. """
        for agent_id, agent in self._agents.items():
            agent.reset()

    def run(self) -> Tuple[StateTrajectory, Dict[int, AgentState], bool, List[Agent]]:
        """ Execute current macro action of ego and forward the state of the environment with collision checking.

        Returns:
            A 4-tuple (trajectory, dict, bool, List[Agent]) giving the new state of the environment, the final
            frame of the simulation, whether the ego has reached its goal, and if it has collided with another
            (possible multiple) agents and if so the colliding agents.
        """
        ego = self._agents[self._ego_id]
        current_observation = Observation(self._initial_frame, self._scenario_map)

        goal_reached = False
        collisions = []

        trajectory = StateTrajectory(self._fps)
        while not goal_reached and not ego.done(current_observation):
            new_frame = {}

            for agent_id, agent in self._agents.items():
                if agent.done(current_observation):
                    agent.alive = False
                if not agent.alive:
                    continue

                new_state = agent.next_state(current_observation)
                new_frame[agent_id] = new_state

                if agent_id == self._ego_id:
                    trajectory.add_state(new_state, reload_path=False)

            current_observation = Observation(new_frame, self._scenario_map)

            collisions = self._check_collisions(ego)
            if collisions: break

            goal_reached = ego.goal.reached(Point(ego.state.position))

        return trajectory, current_observation.frame, goal_reached, collisions

    def _create_agents(self) -> Dict[int, Agent]:
        """ Initialise new agents. Each non-ego is a TrajectoryAgent, while the ego is a MacroAgent. """
        agents = {}
        for aid, state in self._initial_frame.items():
            if aid == self._ego_id:
                agents[aid] = MacroAgent(aid, state, self._metadata[aid], fps=self._fps)
            else:
                agents[aid] = TrajectoryAgent(aid, state, self._metadata[aid], fps=self._fps, open_loop=self._open_loop)
        return agents

    def _check_collisions(self, ego: Agent) -> List[Agent]:
        """ Check for collisions with the given vehicle in the environment. """
        colliding_agents = []
        for agent_id, agent in self._agents.items():
            if agent_id == ego.agent_id:
                continue

            if agent.vehicle.overlaps(ego.vehicle):
                colliding_agents.append(agent)

        return colliding_agents

    def plot(self, axis: plt.Axes = None) -> plt.Axes:
        """ Plot the current agents and the road layout for visualisation purposes.

        Args:
            axis: Axis to draw on
        """
        if axis is None:
            fig, axis = plt.subplots()

        plot_map(self._scenario_map, markings=True, ax=axis)
        for agent_id, agent in self._agents.items():
            vehicle = agent.vehicle
            bounding_box = calculate_rotated_bboxes(vehicle.center[0], vehicle.center[1], vehicle.length, vehicle.width, vehicle.heading)
            pol = plt.Polygon(bounding_box)
            axis.add_patch(pol)

    @property
    def ego_id(self) -> int:
        """ ID of the ego vehicle"""
        return self._ego_id

    @property
    def agents(self) -> Dict[int, Agent]:
        """ Return current agents of the environment """
        return self._agents

    @property
    def initial_frame(self) -> Dict[int, AgentState]:
        """ Return the initial state of the environment """
        return self._initial_frame

    @property
    def fps(self) -> int:
        """ Executing frame rate of the simulator"""
        return self._fps

    @property
    def metadata(self) -> Dict[int, AgentMetadata]:
        """ Metadata of agents in the current frame """
        return self._metadata
