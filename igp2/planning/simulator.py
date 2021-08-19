from typing import Dict

from igp2.agent import AgentState, TrajectoryAgent, Agent, AgentMetadata, MacroAgent
from igp2.goal import Goal
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.trajectory import Trajectory, StateTrajectory


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
                 fps: int = 10):
        """Initialise new light-weight simulator with the given params.

        Args:
            ego_id: ID of the ego vehicle
            initial_frame: initial state of the environment
            metadata: metadata describing the agents in the environment
            scenario_map: current road layout
            fps: frame rate of simulation
        """
        assert ego_id in initial_frame, f"Ego ID {ego_id} is not in the initial frame!"
        assert ego_id in metadata, f"Ego ID {ego_id} not among given metadata!"

        self._scenario_map = scenario_map
        self._ego_id = ego_id
        self._initial_frame = initial_frame
        self._metadata = metadata
        self._fps = fps

        self._agents = self._create_agents()

    def update_trajectory(self, agent_id: int, new_trajectory: Trajectory):
        """ Update the predicted trajectory of the non-ego agent. Has no effect for ego or if agent_id not in agents

        Args:
            agent_id: ID of agent to update
            new_trajectory: new trajectory for agent
        """
        if agent_id in self._agents:
            self._agents[agent_id].trajectory = new_trajectory

    def update_ego_action(self, action: MacroAction):
        """ Update the current macro action of the ego vehicle.

        Args:
            action: new macro action to execute
        """
        self._agents[self._ego_id].update_macro_action(action)

    def update_ego_goal(self, goal: Goal):
        """ Update the final goal of the ego vehicle.

        Args:
            goal: new goal to reach
        """
        self._agents[self._ego_id].update_goal(goal)

    def run(self):
        """ Execute current macro action of ego and forward the state of the environment with collision checking.

        Returns:
            A 3-tuple (trajectory, bool, int) giving the new state of the environment, whether the ego has
            reached its goal, and if it has collided with another agent and if so the ID of the colliding agent.
        """
        t = 0
        done = False
        collision_id = None

        current_frame = self._initial_frame
        ego = self._agents[self._ego_id]
        trajectory = StateTrajectory(self._fps)
        return trajectory, done, collision_id  # TODO: Complete

        while not ego.done(current_frame, self._scenario_map):
            t += 1
            break

        return current_frame, done, collision_id

    def _create_agents(self) -> Dict[int, Agent]:
        """ Initialise new agents. Each non-ego is a TrajectoryAgent, while the ego is a MacroAgent. """
        agents = {}
        for aid in self._initial_frame.keys():
            agent_cls = TrajectoryAgent if aid != self._ego_id else MacroAgent
            agents[aid] = agent_cls(aid, self._metadata[aid])
        return agents

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
