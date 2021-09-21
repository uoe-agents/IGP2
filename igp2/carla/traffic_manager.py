import random
from typing import Callable, Optional

import carla
import numpy as np

from igp2.agent.agent import Agent
from igp2.agent.agentstate import AgentMetadata, AgentState
from igp2.agent.macro_agent import MacroAgent
from igp2.goal import PointGoal


class TrafficManager:
    """ Class that manages non-ego CARLA agents in a synchronous way. The traffic manager manages its own list of
    agents that it synchronises with the CarlaSim object. """

    def __init__(self, n_agents: int = 5, ego: Agent = None):
        """ Initialise a new traffic manager.

        Note:
            This class manages its own list of agents that are then updated through the CarlaSim.take_action()

        Args:
            n_agents: Number of agents to manage
            ego: Optional ego vehicle in the simulation used to determine the spawn area of vehicles.
                If not specified then vehicles may be spawned across the whole map.
         """
        self.__n_agents = n_agents
        self.__ego = ego
        self.__spawn_radius = None
        self.__speed_lims = (2.0, 15.0)
        self.__agents = {}
        self.__enabled = False

    def update(self, simulation):
        """ This method updates the list of managed agents based on their state.
        All vehicles outside the spawn radius are de-spawned."""
        if len(self.__agents) == 0:
            for i in range(self.__n_agents):
                self.__spawn_agent(simulation)
        else:
            ego_position = self.__ego.state.position
            for agent_id, agent in self.__agents.items():
                distance_to_ego = np.linalg.norm(agent.state.position - ego_position)
                if distance_to_ego > self.__spawn_radius:
                    self.__remove_agent(agent, simulation)
                    self.__spawn_agent(simulation)

    def destroy(self, simulation):
        for agent_id in self.__agents:
            simulation.remove_agent(agent_id)

        self.__agents = {}

    def __spawn_agent(self, simulation):
        """Spawn new agents acting as traffic through the given callback function. """
        agent_id = max(simulation.used_ids)
        while agent_id in simulation.used_ids:
            agent_id += 1

        spawn_points = np.array(simulation.world.get_map().get_spawn_points())
        spawn_locations = np.array([[p.location.x, p.location.y] for p in spawn_points])

        blueprint_library = simulation.world.get_blueprint_library()
        blueprint = random.choice(blueprint_library.filter('vehicle.*.*'))

        # Calculate valid spawn points based on spawn radius
        if self.__ego is not None:
            ego_position = self.__ego.state.position
            distances = np.linalg.norm(spawn_locations - ego_position, axis=1)
            valid_spawns = spawn_points[(self.__ego.view_radius <= distances) & (distances <= self.__spawn_radius)]
        else:
            valid_spawns = spawn_points

        # Sample spawn state
        spawn = random.choice(valid_spawns)
        heading = np.deg2rad(spawn.rotation.yaw)
        speed = random.uniform(self.__speed_lims[0], self.__speed_lims[1])
        initial_state = AgentState(time=simulation.timestep,
                                   position=np.array([spawn.location.x, spawn.location.y]),
                                   velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
                                   acceleration=np.array([0.0, 0.0]),
                                   heading=heading)

        # Sample goal for agent
        goal = PointGoal()

        # Spawn the agent
        # TODO: Create custom behaviour agent that also supports driving to the path while also respecting priorities
        agent = MacroAgent(agent_id, initial_state,
                           AgentMetadata(**AgentMetadata.CAR_DEFAULT),
                           goal, simulation.fps)
        self.__agents[agent.agent_id] = agent
        simulation.add_agent(agent, blueprint)

    def __remove_agent(self, agent: Agent, simulation):
        del self.__agents[agent.agent_id]
        simulation.remove_agent(agent)

    def set_agents_count(self, value: int):
        """ Set the number of agents to spawn as traffic. """
        assert value >= 0, f"Number of agents cannot was negative."
        self.__n_agents = value

    def set_ego_agent(self, agent: Agent):
        """ Set an ego agent used for spawn radius calculations in vehicle
        spawning based on the agent's view radius """
        assert hasattr(agent, "view_radius"), f"No view radius given for the ego agent."
        assert agent.view_radius is not None, f"View radius of the given ego agent was None."

        self.__ego = agent
        self.__spawn_radius = 2 * agent.view_radius

    def set_spawn_speed(self, low: float, high: float):
        """ Set the initial spawn speed interval of vehicles."""
        assert low >= 0.0, f"Lower speed bound cannot be negative."
        assert high > low, f"Higher speed limit must be larger than lower speed limit."

        self.__speed_lims = (low, high)

    @property
    def ego(self) -> Agent:
        """ The ID of the ego vehicle in the simulation. """
        return self.__ego

    @property
    def n_agents(self) -> int:
        """ Number of agents to maintain as traffic in the simulation """
        return self.__n_agents

    @property
    def enabled(self) -> bool:
        """Whether the traffic manager is turned on. """
        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool):
        assert isinstance(value, bool)
        self.__enabled = value
