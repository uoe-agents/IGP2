from random import random
from typing import Callable, Optional

import carla

from igp2.agent.agent import Agent
from igp2.agent.agentstate import AgentMetadata
from igp2.agent.macro_agent import MacroAgent
from igp2.agent.mcts_agent import MCTSAgent
from igp2.carla.carla_client import CarlaSim


class TrafficManager:
    """ Class that manages non-ego CARLA agents in a synchronous way. The traffic manager manages its own list of
    agents that is separate from that of the CarlaSim class. """

    def __init__(self, n_agents: int = 5, ego: MCTSAgent = None):
        """ Initialise a new traffic manager. Must be attached to an existing CarlaSim object through
        CarlaSim.set_traffic_manager() to work.

        Note:
            This class manages its own list of agents that are then updated through the CarlaSim.take_action()

        Args:
            n_agents: Number of agents to manage
            ego: Optional ID of the ego vehicle in the simulation used to determine the spawn area of vehicles.
                If not specified then vehicles may be spawned across the whole map.
         """
        self.__n_agents = n_agents
        self.__ego = ego
        self.__simulation: CarlaSim = None
        self.__agents = {}

    def __del__(self):
        self.destroy()

    def update(self):
        if len(self.__agents) == 0:
            self.__spawn_agents()

        for agent_id, agent in self.__agents.items():
            # TODO: Complete traffic manager update code
            pass

    def destroy(self):
        for agent_id in self.__agents:
            self.__simulation.remove_agent(agent_id)

        self.__simulation = None
        self.__agents = {}

    def __spawn_agents(self):
        """Spawn new agents acting as traffic through the given callback function. """
        agent_id = 0
        used_ids = [agent.agent_id for agent in self.__simulation.agents]

        for i in range(self.__n_agents):
            while agent_id in used_ids:
                agent_id += 1
            # TODO
            blueprint_library = self.__simulation.world.get_blueprint_library()
            blueprint = random.choice(blueprint_library.filter('vehicle.*.*'))
            agent = MacroAgent(agent_id, state, AgentMetadata(**AgentMetadata.CAR_DEFAULT), goal, self.__simulation.fps)
            self.agents[agent.agent_id] = agent
            self.__simulation.add_agent(agent, blueprint)

    @property
    def ego(self) -> int:
        """ The ID of the ego vehicle in the simulation. """
        return self._ego_id
