from typing import Dict

from igp2.agent import AgentState, TrajectoryAgent, Agent, AgentMetadata, MacroAgent
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.trajectory import Trajectory


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
        self._current_frame = initial_frame
        self._metadata = metadata
        self._fps = fps

        self._agents = self._create_agents()

    def update_trajectory(self, agent_id: int, new_trajectory: Trajectory):
        """ Update the predicted trajectories of non-ego agents. Has no effect for ego or if agent_id not in agents

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
        self._agents[self._ego_id]

    def run(self):
        """ Execute current macro action of ego and forward the state of the environment with collision checking.

        Returns:
            A 4-tuple (frame, int, bool, int) giving the new state of the environment, the number of time steps
            elapsed, whether the ego has reached its goal, and if it has collided with another agent and if so the ID
            of the colliding agent.
        """
        t = 0
        done = False
        ego = self._agents[self._ego_id]
        while not ego.done(self._current_frame, self._scenario_map):
            t += 1
            break

        return self._current_frame, t, done, None

    def _create_agents(self) -> Dict[int, Agent]:
        """ Initialise new agents. Each non-ego is a TrajectoryAgent, while the ego is a MacroAgent. """
        agents = {
            aid: TrajectoryAgent(aid, self._metadata[aid])
            for aid in self._current_frame.keys()
        }
        agents[self._ego_id] = MacroAgent()
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
    def current_frame(self) -> Dict[int, AgentState]:
        """ Return current state of the environment stored in the simulator """
        return self._current_frame

    @current_frame.setter
    def current_frame(self, value: Dict[int, AgentState]):
        """ Overwrite current frame of the environment with new frame"""
        self._current_frame = value

    @property
    def fps(self) -> int:
        """ Executing frame rate of the simulator"""
        return self._fps

    @property
    def metadata(self) -> Dict[int, AgentMetadata]:
        """ Metadata of agents in the current frame """
        return self._metadata
