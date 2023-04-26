from collections import defaultdict
import logging
from typing import Dict, List

from igp2.opendrive.map import Map
from igp2.agents.agent import Agent
from igp2.core.vehicle import Action, Observation

logger = logging.getLogger(__name__)


class Simulation:
    """ A lightweight simulator for IGP2 to perform rapid testing. """

    def __init__(self,
                 scenario_map: Map,
                 fps: int = 20,
                 open_loop: bool = False):
        """ Initialise new simulation.

        Args:
            scenario_map: The current road layout.
            fps: Execution frame-rate.
            open_loop: If true then no physical controller will be applied.
        """
        self.__scenario_map = scenario_map
        self.__fps = fps
        self.__open_loop = open_loop

        self.__t = 0
        self.__state = {}
        self.__agents = {}
        self.__actions = defaultdict(list)

    def add_agent(self, new_agent: Agent, rolename: str = None):
        """ Add a new agent to the simulation.

        Args:
            new_agent: Agent to add.
            rolename: Currently unused. Optional string to describe role of the vehicle.
        """
        if new_agent.agent_id in self.__agents \
                and self.__agents[new_agent.agent_id] is not None:
            raise ValueError(f"Agent with ID {new_agent.agent_id} already exists.")

        self.__agents[new_agent.agent_id] = new_agent
        self.__state[new_agent.agent_id] = new_agent.vehicle.get_state(0)
        logger.debug(f"Added Agent {new_agent.agent_id}")

    def remove_agent(self, agent_id: int):
        """ Remove an agent from the simulation.

        Args:
            agent_id: Agent ID to remove.
        """
        self.__agents[agent_id].alive = False
        self.__agents[agent_id] = None
        logger.debug(f"Removed Agent {agent_id}")

    def reset(self):
        """ Remove all agents and reset internal state of simulation. """
        self.__t = 0
        self.__agents = {}
        self.__state = {}

    def step(self):
        """ Advance simulation by one time step. """
        logger.debug(f"Simulation step {self.__t}")
        self.__take_actions()
        self.__t += 1

    def __take_actions(self):
        new_frame = {}
        observation = Observation(self.__state, self.__scenario_map)

        for agent_id, agent in self.__agents.items():
            if agent is None or not agent.alive:
                continue
            if not agent.alive or self.__t > 0 and agent.done(observation):
                self.remove_agent(agent_id)
                continue

            new_state, action = agent.next_state(observation, return_action=True)

            agent.trajectory_cl.add_state(new_state, reload_path=False)
            self.__actions[agent_id].append(action)
            new_frame[agent_id] = new_state

            agent.alive = len(self.__scenario_map.roads_at(new_state.position)) > 0

        self.__state = new_frame

    @property
    def scenario_map(self) -> Map:
        """ The road layout of the simulation. """
        return self.__scenario_map

    @property
    def agents(self) -> Dict[int, Agent]:
        """ Agents in the simulation, mapping agent IDs to agents. """
        return self.__agents

    @property
    def actions(self) -> Dict[int, List[Action]]:
        """ List of actions (acceleration and steering) taken by every vehicle. """
        return self.__actions

    @property
    def t(self) -> int:
        """ The current time step of the simulation. """
        return self.__t
