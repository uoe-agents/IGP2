import random
import logging
from typing import List, Dict

import carla
import numpy as np

from igp2.opendrive.map import Map
from igp2.agents.agent import Agent
from igp2.agents.traffic_agent import TrafficAgent
from igp2.core.vehicle import Observation
from igp2.core.agentstate import AgentState
from igp2.core.goal import PointGoal
from igp2.carlasim import CarlaAgentWrapper, get_actor_blueprints

logger = logging.getLogger(__name__)


class TrafficManager:
    """ Class that manages non-ego CARLA agents in a synchronous way. The traffic manager manages its own list of
    agents that it synchronises with the CarlaSim object. """

    def __init__(self,
                 scenario_map: Map,
                 n_agents: int = 5,
                 ego: Agent = None,
                 spawn_tries: int = 10):
        """ Initialise a new traffic manager.

        Note:
            This class manages its own list of agents that are then updated through the CarlaSim.take_action()

        Args:
            scenario_map: The current road layout
            n_agents: Number of agents to manage
            ego: Optional ego vehicle in the simulation used to determine the spawn area of vehicles.
                If not specified then vehicles may be spawned across the whole map.
            spawn_tries: How many number of times to try spawn a single vehicle.
         """
        self.__scenario_map = scenario_map
        self.__n_agents = n_agents
        self.__ego = ego
        self.__spawn_radius = None
        self.__agents = {}
        self.__enabled = False
        self._max_spawn_tries = spawn_tries
        self.__spawns = []
        self._actor_filter = "vehicle.*"
        self._actor_generation = "2"

    def update(self, simulation, observation: Observation = None):
        """ This method updates the list of managed agents based on their state.
        All vehicles outside the spawn radius are de-spawned.

        Args:
            simulation: The currently running simulation object
            observation: The last observation of the environment
        """
        if not self.enabled:
            return

        for agent_id, agent in self.__agents.items():
            if agent is None:
                continue

            if self.__ego is not None:
                ego_position = self.__ego.state.position
                distance_to_ego = np.linalg.norm(agent.state.position - ego_position)
                if distance_to_ego > self.__spawn_radius:
                    self.__remove_agent(agent, simulation)
                    continue

            if observation is not None and agent.done(observation):
                self.__find_destination(agent, agent.state)

        agents_existing = len([agent for agent in self.__agents.values() if agent is not None])
        if agents_existing < self.__n_agents:
            for i in range(self.__n_agents - agents_existing):
                self.__spawn_agent(simulation)
        simulation.world.tick()

    def disable(self, simulation):
        """ Disable the traffic manager, removing all managed vehicles from the simulation. """
        self.__enabled = False
        for agent_id, agent in self.__agents.items():
            if agent is not None:
                self.__remove_agent(agent_id, simulation)
        self.__agents = {}

    def __spawn_agent(self, simulation):
        """Spawn new agents acting as traffic through the given callback function. """
        spawn_points = np.array(self.spawns)
        spawn_locations = np.array([[p.location.x, p.location.y] for p in spawn_points])

        blueprint = self.__random_blueprint(simulation)

        # Calculate valid spawn points based on spawn radius
        valid_spawns = spawn_points
        if self.__ego is not None:
            ego_position = self.__ego.state.position
            ego_position[1] *= -1
            distances = np.linalg.norm(spawn_locations - ego_position, axis=1)
            valid_spawns = spawn_points[(self.__ego.view_radius <= distances) & (distances <= self.__spawn_radius)]

        # Sample spawn state and spawn actor
        try_count = 0
        while try_count < self._max_spawn_tries:
            spawn = random.choice(valid_spawns)
            spawn.location.z += 0.5
            spawn.rotation.roll = 0.0
            spawn.rotation.pitch = 0.0
            heading = np.deg2rad(-spawn.rotation.yaw)
            try:
                vehicle = simulation.world.spawn_actor(blueprint, spawn)
                break
            except:
                try_count += 1
        else:
            logger.debug("Couldn't spawn traffic vehicle!")
            return

        # Create agent and set properties
        initial_state = AgentState(time=simulation.timestep,
                                      position=np.array([spawn.location.x, -spawn.location.y]),
                                      velocity=np.array([0.001 * np.cos(heading), 0.001 * np.sin(heading)]),
                                      acceleration=np.array([0.0, 0.0]),
                                      heading=heading)
        agent = TrafficAgent(vehicle.id, initial_state, fps=simulation.fps)
        agent = CarlaAgentWrapper(agent, vehicle)

        self.__find_destination(agent, initial_state)

        # Wrap agent for CARLA control
        self.__agents[agent.agent_id] = agent
        simulation.agents[agent.agent_id] = agent

        logger.debug(f"Traffic agent {agent.agent_id} (actor {agent.actor_id}) spawned at {spawn.location}.")

    def __find_destination(self, agent_wrapper: CarlaAgentWrapper, state: AgentState):
        agent = agent_wrapper.agent
        # agent_wrapper.reset_waypoints()

        destination = random.choice(self.spawns).location
        goal = PointGoal(np.array([destination.x, -destination.y]), 1.0)
        agent.set_destination(Observation({agent.agent_id: state}, self.__scenario_map), goal)

        logger.debug(f"Destination set to {goal} for Agent {agent.agent_id}")

    def __remove_agent(self, agent_wrapper: CarlaAgentWrapper, simulation):
        self.__agents[agent_wrapper.agent_id] = None
        simulation.remove_agent(agent_wrapper.agent_id)

    def __random_blueprint(self, simulation) -> carla.ActorBlueprint:
        """ Get a random blueprint for a TrafficAgent"""
        blueprint = random.choice(get_actor_blueprints(simulation.world, self._actor_filter, self._actor_generation))
        # blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        return blueprint

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
        self.__spawn_radius = 1.5 * agent.view_radius

    def set_spawn_filter(self, actor_filter: str):
        """ Set what types of actors to spawn. """
        self._actor_filter = actor_filter

    def set_spawn_generation(self, actor_generation: str):
        """ Set which version of actor blueprint generation to use. This is usually set to 2.
        Must be either '1', '2', or 'All'. """
        assert actor_generation in ["1", "2", "All"], "Invalid actor generation type given. "
        self._actor_generation = actor_generation

    # def set_agent_behaviour(self, value: str = "normal"):
    #     """ Set the behaviour of all agents as given by the behaviour types.
    #     If set to random, then each vehicle will be randomly assigned a behaviour type. """
    #     pass

    @property
    def ego(self) -> Agent:
        """ The ID of the ego vehicle in the simulation. """
        return self.__ego

    @property
    def agents(self) -> Dict[int, Agent]:
        """ The agents managed by the manager"""
        return self.__agents

    @property
    def n_agents(self) -> int:
        """ Number of agents to maintain as traffic in the simulation """
        return self.__n_agents

    @property
    def spawns(self) -> List[carla.Transform]:
        """ List of all possible spawn points"""
        return self.__spawns

    @spawns.setter
    def spawns(self, value: List[carla.Transform]):
        self.__spawns = value

    @property
    def enabled(self) -> bool:
        """Whether the traffic manager is turned on. """
        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool):
        assert isinstance(value, bool)
        self.__enabled = value
