import time
from typing import Union, List
import os
import numpy as np
import carla
import subprocess
import psutil
import platform
import logging

from carla import Transform, Location, Rotation, Vector3D
from igp2.agents.agent import Agent
from igp2.agents.agentstate import AgentState
from igp2.carla.traffic_manager import TrafficManager
from igp2.opendrive.map import Map
from igp2.vehicle import Observation

logger = logging.getLogger(__name__)


class CarlaSim:
    """ An interface to the CARLA simulator """
    TIMEOUT = 10.0
    MAX_ACCELERATION = 5

    def __init__(self, fps=20, xodr=None, port=2000, carla_path='/opt/carla-simulator'):
        """ Launch the CARLA simulator and define a CarlaSim object, which keeps the connection to the CARLA
        server, manages agents and advances the environment.

        Note:
             The first agent of type MCTSAgent to be added to the simulation will be treated as the ego vehicle.

        Args:
            fps: number of frames simulated per second
            xodr: path to a .xodr OpenDrive file defining the road layout
            port: port to use for communication with the CARLA simulator
            carla_path: path to the root directory of the CARLA simulator
        """
        self.scenario_map = Map.parse_from_opendrive(xodr)
        self.__carla_process = None
        if "CarlaUE4.exe" not in [p.name() for p in psutil.process_iter()]:
            sys_name = platform.system()
            if sys_name == "Windows":
                args = [os.path.join(carla_path, 'CarlaUE4.exe'), '-quality-level=Low', f'-carla-rpc-port={port}']
            elif sys_name == "Linux":
                args = [os.path.join(carla_path, 'CarlaUE4.sh'), '-quality-level=Low', f'-carla-rpc-port={port}']
            self.__carla_process = subprocess.Popen(args)

        self.__port = port
        self.__client = carla.Client('localhost', port)
        self.__client.set_timeout(self.TIMEOUT)  # seconds
        self.__wait_for_server()
        if xodr is not None:
            self.load_opendrive_world(xodr)

        self.__fps = fps
        self.__timestep = 0

        self.__world = self.__client.get_world()
        settings = self.__world.get_settings()
        settings.fixed_delta_seconds = 1 / fps
        settings.synchronous_mode = True
        self.__world.apply_settings(settings)

        self.agents = {}
        self.__dead_agent_ids = []
        self.__actor_ids = {}

        self.__spectator = self.__world.get_spectator()
        self.__traffic_manager = TrafficManager()

    def __del__(self):
        if self.__carla_process is None:
            return

        proc = self.__carla_process
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()

    def load_opendrive_world(self, xodr: str):
        with open(xodr, 'r') as f:
            opendrive = f.read()
        self.__client.set_timeout(60.0)
        self.__client.generate_opendrive_world(opendrive)
        self.__client.set_timeout(self.TIMEOUT)

    def step(self):
        """ Advance the simulation by one time step"""
        if self.__traffic_manager.enabled:
            self.__traffic_manager.update(self)

        self.__world.tick()
        self.__timestep += 1

        if self.__traffic_manager.enabled:
            self.__traffic_manager.take_actions(self)

        frame = self.__get_current_frame()
        observation = Observation(frame, self.scenario_map)
        self.__take_actions(observation)

    def add_agent(self, agent: Agent, blueprint: carla.ActorBlueprint = None) -> carla.Actor:
        """ Add a vehicle to the simulation. Defaults to an Audi A2 for blueprints if not explicitly given.

        Args:
            agent: agent to add
            blueprint: Optional blueprint defining the properties of the actor

        Returns:
            The newly added actor
        """
        if blueprint is None:
            blueprint_library = self.__world.get_blueprint_library()
            blueprint = blueprint_library.find('vehicle.audi.a2')

        state = agent.state
        yaw = np.rad2deg(state.heading)
        transform = Transform(Location(x=state.position[0], y=-state.position[1], z=0.1), Rotation(yaw=yaw))
        actor = self.__world.spawn_actor(blueprint, transform)
        actor.set_target_velocity(Vector3D(state.velocity[0], -state.velocity[1], 0.))
        self.__actor_ids[agent.agent_id] = actor.id
        self.agents[agent.agent_id] = agent
        return actor

    def remove_agent(self, agent: Union[Agent, int]):
        """ Remove the given agent from the simulation.

        Args:
            agent: Either the instance of an Agent or an agent ID.
        """
        actor_list = self.__world.get_actors()
        if isinstance(agent, Agent):
            agent_id = agent.agent_id
        elif isinstance(agent, int):
            agent_id = agent
        else:
            raise TypeError(f"Not an Agent instance or an agent ID specified for removal! Object was: {agent}")

        actor = actor_list.find(self.__actor_ids[agent_id])
        actor.destroy()
        self.agents[agent_id].alive = False
        self.__dead_agent_ids.append(agent_id)

    def get_traffic_manager(self) -> "TrafficManager":
        """ Enables and returns the internal traffic manager of the simulation."""
        self.__traffic_manager.enabled = True

        spawn_points = []
        for p in self.__world.get_map().get_spawn_points():
            distances = [np.isclose((q.location - p.location).x, 0.0) and
                         np.isclose((q.location - p.location).y, 0.0) for q in spawn_points]
            if not any(distances):
                spawn_points.append(p)
        self.__traffic_manager.spawns = spawn_points
        return self.__traffic_manager

    def attach_spectator(self, actor: carla.Actor, offset: float = -np.pi / 2):
        """ Attach the spectator (view) of CARLA to a given actor at the given angle offset.

         Args:
             actor: Actor to attach the spectator to
             offset: Angle offset of the viewpoint
         """
        # TODO: Debug
        def get_transform(vehicle_location: carla.Location, angle: float, d: float = 20):
            location = carla.Location(d * np.cos(angle), d * np.sin(angle), 2.0) + vehicle_location
            return carla.Transform(location, carla.Rotation(yaw=180 + np.rad2deg(angle), pitch=-15))

        self.__spectator.set_transform(get_transform(actor.get_location(), offset))

    def run(self, steps=400):
        """ Run the simulation for a number of time steps """
        for i in range(steps):
            logger.debug(f"CARLA step {i} of {steps}.")
            self.step()
            time.sleep(1 / self.__fps)

    def __take_actions(self, observation: Observation):
        actor_list = self.__world.get_actors()

        commands = []
        for agent_id, agent in list(self.agents.items()):
            if agent_id not in self.__dead_agent_ids:
                action = agent.next_action(observation)
                if action is None or agent.done(observation):
                    self.remove_agent(agent)
                    continue

                control = carla.VehicleControl()
                norm_acceleration = action.acceleration / self.MAX_ACCELERATION
                if action.acceleration >= 0:
                    control.throttle = min(1., norm_acceleration)
                    control.brake = 0.
                else:
                    control.throttle = 0.
                    control.brake = min(-norm_acceleration, 1.)
                control.steer = -action.steer_angle
                control.hand_brake = False
                control.manual_gear_shift = False

                actor = actor_list.find(self.__actor_ids[agent_id])
                command = carla.command.ApplyVehicleControl(actor, control)
                commands.append(command)
        self.__client.apply_batch_sync(commands)

    def __get_current_frame(self):
        actor_list = self.__world.get_actors()
        frame = {}
        for agent_id in self.agents:
            if agent_id not in self.__dead_agent_ids:
                actor = actor_list.find(self.__actor_ids[agent_id])
                transform = actor.get_transform()
                heading = np.deg2rad(-transform.rotation.yaw)
                velocity = actor.get_velocity()
                acceleration = actor.get_acceleration()
                state = AgentState(time=self.__timestep,
                                   position=np.array([transform.location.x, -transform.location.y]),
                                   velocity=np.array([velocity.x, -velocity.y]),
                                   acceleration=np.array([acceleration.x, -acceleration.x]),
                                   heading=heading)
                frame[agent_id] = state
        return frame

    def __wait_for_server(self):
        for i in range(10):
            try:
                self.__client = carla.Client('localhost', self.__port)
                self.__client.set_timeout(self.TIMEOUT)  # seconds
                self.__client.get_world()
                return
            except RuntimeError:
                pass
        self.__client.get_world()

    @property
    def client(self) -> carla.Client:
        """ The CARLA client to the server. """
        return self.__client

    @property
    def world(self) -> carla.World:
        """The current CARLA world"""
        return self.__world

    @property
    def spectator(self) -> carla.Actor:
        """ The spectator camera of the world. """
        return self.__spectator

    @property
    def fps(self) -> int:
        """ Execution frequency of the simulation. """
        return self.__fps

    @property
    def timestep(self) -> int:
        """ Current timestep of the simulation"""
        return self.__timestep

    @property
    def dead_ids(self) -> List[int]:
        """ List of Agent IDs that have been used previously during the simulation"""
        return self.__dead_agent_ids

    @property
    def used_ids(self) -> List[int]:
        """ List of Agent IDs which are either in use or have been used already during simulation. """
        return [agent_id for agent_id in self.agents] + self.__dead_agent_ids
