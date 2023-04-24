import igp2 as ip
from typing import List

from igp2.agents.agent import Agent


class ManeuverAgent(Agent):
    """ For testing purposes. Agent that executes a sequence of maneuvers"""

    def __init__(self,
                 maneuver_configs: List[ip.ManeuverConfig],
                 agent_id: int,
                 initial_state: ip.AgentState,
                 fps: int = 20,
                 view_radius: float = None):
        super().__init__(agent_id, initial_state, view_radius)
        self._vehicle = ip.TrajectoryVehicle(initial_state, initial_state.metadata, fps)
        self.maneuver_configs = maneuver_configs
        self.maneuver = None

    def create_next_maneuver(self, agent_id, observation):
        if len(self.maneuver_configs) > 0:
            config = self.maneuver_configs.pop(0)
            config.config_dict["fps"] = self.fps
            self.maneuver = ip.CLManeuverFactory.create(
                config, agent_id, observation.frame, observation.scenario_map)
        else:
            self.maneuver = None

    def next_action(self, observation: ip.Observation = None) -> ip.Action:
        if self.maneuver is None or self.maneuver.done(observation):
            self.create_next_maneuver(self.agent_id, observation)

        if self.maneuver is None:
            return ip.Action(0., 0.)
        else:
            return self.maneuver.next_action(observation)

    def done(self, observation: ip.Observation) -> bool:
        return False
