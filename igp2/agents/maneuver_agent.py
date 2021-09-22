from typing import List

from igp2.agents.agent import Agent
from igp2.agents.agentstate import AgentMetadata, AgentState
from igp2.planlibrary.maneuver import ManeuverConfig
from igp2.planlibrary.maneuver_cl import CLManeuverFactory
from igp2.vehicle import Observation, Action, TrajectoryVehicle


class ManeuverAgent(Agent):
    """ For testing purposes. Agent that executes a sequence of maneuvers"""

    def __init__(self, maneuver_configs: List[ManeuverConfig], agent_id: int,
                 initial_state: AgentState, agent_metadata: AgentMetadata, fps: int = 20,
                 view_radius: float = None, ):
        super().__init__(agent_id, initial_state, agent_metadata, view_radius)
        self._vehicle = TrajectoryVehicle(initial_state, agent_metadata, fps)
        self.maneuver_configs = maneuver_configs
        self.maneuver = None

    def create_next_maneuver(self, agent_id, observation):
        if len(self.maneuver_configs) > 0:
            config = self.maneuver_configs.pop(0)
            self.maneuver = CLManeuverFactory.create(config, agent_id, observation.frame,
                                                     observation.scenario_map)
        else:
            self.maneuver = None

    def next_action(self, observation: Observation = None) -> Action:
        if self.maneuver is None or self.maneuver.done(observation):
            self.create_next_maneuver(self.agent_id, observation)

        if self.maneuver is None:
            return Action(0., 0.)
        else:
            return self.maneuver.next_action(observation)

    def done(self, observation: Observation) -> bool:
        return False
