from typing import Dict

from igp2.agent import AgentState, TrajectoryAgent, Agent
from igp2.opendrive.map import Map
from igp2.recognition.goalprobabilities import GoalsProbabilities


class Simulator:
    """ Lightweight environment simulator useful for rolling out scenarios in MCTS.

    One agent is designated as the ego vehicle, while the other agents follow predefined trajectories calculated
    during goal recognition. Simulation is performed at a given frequency with collision checking.
    """

    def __init__(self,
                 ego_id: int,
                 initial_frame: Dict[int, AgentState],
                 scenario_map: Map):
        self._scenario_map = scenario_map
        self._ego_id = ego_id
        self._current_frame = initial_frame

        self._agents = self._create_agents()

    def update_agent_trajectories(self, predictions: GoalsProbabilities):
        """ Update the predicted trajectories of non-ego agents.

        Args:
            predictions: goal probability and trajectory predictions
        """

    def _create_agents(self) -> Dict[int, Agent]:
        """ Initialise new agents. Each non-ego is a TrajectoryAgent, while the ego is a MacroAgent. """
        agents = {aid: TrajectoryAgent(aid, ) for aid in self._current_frame.keys()}
        agents[self._ego_id] = None  # TODO: Create MacroAgent()
        return agents
