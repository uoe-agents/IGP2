from collections import defaultdict
import logging
import numpy as np
from typing import Dict, List

from igp2.opendrive.map import Map
from igp2.agents.agent import Agent
from igp2.core.vehicle import Action, Observation
from igp2.core.agentstate import AgentState

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
        self.__state[agent_id] = None
        logger.debug(f"Removed Agent {agent_id}")

    def reset(self):
        """ Remove all agents and reset internal state of simulation. """
        self.__t = 0
        self.__agents = {}
        self.__state = {}

    def step(self) -> bool:
        """ Advance simulation by one time step.

        Returns:
            True if any agent is still alive else False.
        """
        if 0 in self.__state:
            logger.info(f"Simulation step {self.__t} - "
                        f"Pos: {np.round(self.__state[0].position, 2)} - "
                        f"Vel: {np.round(self.__state[0].speed, 2)} - "
                        f"Mcr: {self.__state[0].macro_action} - "
                        f"Man: {self.__state[0].maneuver}")
        else:
            logger.info(f"Simulation step {self.__t}")
        logger.info(f"N agents: {len(self.__agents)}")
        alive = self.take_actions()
        return alive

    def take_actions(self, action: Action = None) -> bool:
        """ Take actions for all agents in the simulation.

        Args:
            action: Optional action to apply to the ego agent (Agent ID 0).
        """
        new_frame = {}

        for agent_id, agent in self.__agents.items():
            if agent is None:
                continue

            observation = self.get_observations(agent_id)

            if not agent.alive:
                self.remove_agent(agent_id)
                continue
            if self.__t > 0 and agent.done(observation):
                agent.alive = False
                continue

            if action is not None and agent_id == 0:
                agent.vehicle.execute_action(action, self.__state[0])
                new_state = agent.vehicle.get_state(observation.frame[agent_id].time + 1)
                new_state.macro_action = str(agent.current_macro)
                new_state.maneuver = str(agent.current_macro.current_maneuver)
            else:
                new_state, action = agent.next_state(observation, return_action=True)

            agent.trajectory_cl.add_state(new_state, reload_path=False)
            self.__actions[agent_id].append(action)
            new_frame[agent_id] = new_state

            on_road = len(self.__scenario_map.roads_at(new_state.position)) > 0
            if not on_road:
                logger.debug(f"Agent {agent_id} went off-road.")

            collision = any(agent.bounding_box.overlaps(ag.bounding_box) for aid, ag in
                            self.__agents.items() if aid != agent_id and ag is not None)
            if collision:
                logger.debug(f"Agent {agent_id} collided with another agent(s)")

            agent.alive = on_road and not collision

        self.__state = new_frame
        self.__t += 1
        return any([agent.alive if agent is not None else False for agent in self.__agents.values()])

    def get_observations(self, agent_id: int = 0):
        """ Get observations for the given agent. Can be overridden to add occlusions to the environment for example.

        Args:
            agent_id: The ID of the agent for which to retrieve observations.
        """
        return Observation(self.__state, self.__scenario_map)

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

    @property
    def state(self) -> Dict[int, AgentState]:
        """ Current joint state of the simulation. """
        return self.__state
