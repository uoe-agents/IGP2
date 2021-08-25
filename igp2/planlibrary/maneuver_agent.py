from typing import List

from igp2.agent import Agent
from igp2.agentstate import AgentMetadata
from igp2.planlibrary.maneuver import ManeuverConfig, CLManeuverFactory
from igp2.vehicle import Observation, Action

class ManeuverAgent(Agent):
    """ For testing purposes. Agent that executes a sequence of maneuvers"""

    def __init__(self, maneuver_configs: List[ManeuverConfig], agent_id: int, agent_metadata: AgentMetadata, view_radius: float = None,):
        super().__init__(agent_id, agent_metadata, view_radius)
        self.maneuver_configs = maneuver_configs
        self.maneuver = None

    def create_next_maneuver(self, agent_id, frame, scenario_map):
        if len(self.maneuver_configs) > 0:
            config = self.maneuver_configs.pop(0)
            self.maneuver = CLManeuverFactory.create(config, agent_id, frame, scenario_map)
        else:
            self.maneuver = None

    def next_action(self, observation: Observation = None) -> Action:
        if self.maneuver is None or self.maneuver.completed(self.agent_id, observation.frame, observation.scenario_map):
            self.create_next_maneuver(self.agent_id, observation.frame, observation.scenario_map)

        if self.maneuver is None:
            return Action(0., -1.)
        else:
            return self.maneuver.next_action(self.agent_id, observation.frame, observation.scenario_map)

    def done(self):
        return False