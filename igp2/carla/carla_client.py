import time
from typing import List, Dict, Optional
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import carla
import subprocess
import psutil
import platform
import logging

from carla import Transform, Location, Rotation, Vector3D
import igp2 as ip

logger = logging.getLogger(__name__)


class CarlaSim:
    """ An interface to the CARLA simulator """
    TIMEOUT = 20.0

    def __init__(self,
                 fps: int = 20,
                 xodr: str = None,
                 map_name: str = None,
                 server: str = "localhost",
                 port: int = 2000,
                 carla_path: str = "/opt/carla-simulator",
                 record: bool = False,
                 rendering: bool = True):
        """ Launch the CARLA simulator and define a CarlaSim object, which keeps the connection to the CARLA
        server, manages agents and advances the environment.

        Note:
             The first agent of type MCTSAgent to be added to the simulation will be treated as the ego vehicle.

        Args:
            fps: number of frames simulated per second
            xodr: path to a .xodr OpenDrive file defining the road layout
            port: port to use for communication with the CARLA simulator
            carla_path: path to the root directory of the CARLA simulator
            rendering: controls whether graphics are rendered
        """
        self.scenario_map = ip.Map.parse_from_opendrive(xodr)
        self.__carla_process = None
        sys_name = platform.system()
        if sys_name == "Windows":
            if "CarlaUE4.exe" not in [p.name() for p in psutil.process_iter()]:
                args = [os.path.join(carla_path, 'CarlaUE4.exe'), '-quality-level=Low',
                        f'-carla-rpc-port={port}', '-dx11']
                self.__carla_process = subprocess.Popen(args)
        elif sys_name == "Linux":
            if "CarlaUE4.sh" not in [p.name() for p in psutil.process_iter()]:
                args = [os.path.join(carla_path, 'CarlaUE4.sh'), '-quality-level=Low', f'-carla-rpc-port={port}']
                self.__carla_process = subprocess.Popen(args)
        else:
            raise RuntimeError("Unsupported system!")

        self.__record = record
        self.__port = port
        self.__client = carla.Client(server, port)
        self.__client.set_timeout(self.TIMEOUT)  # seconds
        self.__wait_for_server()
        if map_name is not None:
            self.__client.load_world(map_name)
        elif xodr is not None:
            self.load_opendrive_world(xodr)

        self.__fps = fps
        self.__timestep = 0

        self.__world = self.__client.get_world()
        self.__original_settings = self.__world.get_settings()
        settings = self.__world.get_settings()
        settings.fixed_delta_seconds = 1 / fps
        settings.synchronous_mode = True
        settings.no_rendering_mode = not rendering
        self.__world.apply_settings(settings)

        self.__record_path = None
        if self.__record:
            now = datetime.now()
            log_name = now.strftime("%d-%m-%Y_%H-%M-%S") + ".log"
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = Path(dir_path)
            repo_path = str(path.parent.parent)
            self.__record_path = os.path.join(repo_path, "scripts", "experiments", "data", "carla_recordings", log_name)
            logger.info(f"Recording simulation under path: {self.__client.start_recorder(self.__record_path, True)}")

        self.agents: Dict[int, ip.carla.CarlaAgentWrapper] = {}

        self.__spectator = self.__world.get_spectator()
        self.__spectator_parent = None
        self.__spectator_transform = None

        self.__traffic_manager = ip.carla.TrafficManager(self.scenario_map)
        self.__warmed_up = False

        self.__world.tick()

    def __del__(self):
        self.__world.apply_settings(self.__original_settings)

        if self.__record:
            self.__client.stop_recorder()

        if self.__carla_process is None:
            return

        proc = self.__carla_process
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()

    def run(self, steps=400):
        """ Run the simulation for a number of time steps """
        for i in range(steps):
            self.step()
            time.sleep(1 / self.__fps)

    def step(self, tick: bool = True):
        """ Advance the simulation by one time step"""
        logger.debug(f"CARLA step {self.__timestep}.")

        if tick:
            self.__world.tick()

        if not self.__warmed_up:
            self.__warm_up()

        self.__timestep += 1
        
        observation = self.__get_current_observation()
        self.__take_actions(observation)
        self.__traffic_manager.update(self, observation)
        self.__update_spectator()

    def add_agent(self,
                  agent: ip.Agent,
                  rolename: str = None,
                  blueprint: carla.ActorBlueprint = None):
        """ Add a vehicle to the simulation. Defaults to an Audi A2 for blueprints if not explicitly given.

        Args:
            agent: Agent to add.
            rolename: Unique name for the actor to spawn.
            blueprint: Optional blueprint defining the properties of the actor.

        Returns:
            The newly added actor.
        """
        if blueprint is None:
            blueprint_library = self.__world.get_blueprint_library()
            blueprint = blueprint_library.find('vehicle.audi.a2')

        if rolename is not None:
            blueprint.set_attribute('role_name', rolename)

        state = agent.state
        yaw = np.rad2deg(-state.heading)
        transform = Transform(Location(x=state.position[0], y=-state.position[1], z=0.1), Rotation(yaw=yaw))
        actor = self.__world.spawn_actor(blueprint, transform)
        # actor.set_target_velocity(Vector3D(state.velocity[0], -state.velocity[1], 0.))

        carla_agent = ip.carla.CarlaAgentWrapper(agent, actor)
        self.agents[carla_agent.agent_id] = carla_agent

    def remove_agent(self, agent_id: int):
        """ Remove the given agent from the simulation.

        Args:
            agent_id: The ID of the agent to remove
        """
        actor = self.agents[agent_id].actor
        actor.destroy()
        self.agents[agent_id].agent.alive = False
        self.agents[agent_id] = None

    def get_traffic_manager(self) -> "TrafficManager":
        """ Enables and returns the internal traffic manager of the simulation."""
        self.__traffic_manager.enabled = True

        spawn_points = []
        for p in self.__world.get_map().get_spawn_points():
            distances = [np.isclose((q.location - p.location).x, 0.0) and
                         np.isclose((q.location - p.location).y, 0.0) for q in spawn_points]
            if not any(distances) and len(self.scenario_map.roads_at((p.location.x, -p.location.y), drivable=True)) > 0:
                spawn_points.append(p)
        self.__traffic_manager.spawns = spawn_points
        return self.__traffic_manager

    def load_opendrive_world(self, xodr: str):
        with open(xodr, 'r') as f:
            opendrive = f.read()
        self.__client.set_timeout(60.0)
        self.__client.generate_opendrive_world(opendrive)
        self.__client.set_timeout(self.TIMEOUT)

    def attach_camera(self,
                      actor: carla.Actor,
                      transform: carla.Transform = carla.Transform(carla.Location(x=1.6, z=1.7))):
        """ Attach a camera to the back of the given actor in third-person view.

        Args:
            actor: The actor to follow
            transform: Optional transform to set the position of the camera
        """
        self.__spectator_parent = actor
        self.__spectator_transform = transform

    def get_ego(self, ego_name: str = "ego") -> Optional["ip.carla.CarlaAgentWrapper"]:
        """ Returns the ego agent if it exists. """
        for agent in self.agents.values():
            if agent is not None and agent.name == ego_name:
                return agent
        return None

    def __update_spectator(self):
        if self.__spectator_parent is not None and self.__spectator_transform is not None:
            actor_transform = self.__spectator_parent.get_transform()
            actor_transform.location += self.__spectator_transform.location
            # actor_transform.rotation += self.__spectator_transform.rotation
            self.__spectator.set_transform(actor_transform)

    def __warm_up(self):
        transforms = {aid: agent.actor.get_transform() for aid, agent in self.agents.items()}
        while True:
            control = carla.VehicleControl(throttle=0.5)
            commands = []
            for agent_id, agent in self.agents.items():
                vel = agent.actor.get_velocity()
                speed = np.sqrt(vel.x ** 2 + vel.y ** 2)
                agent.actor.set_transform(transforms[agent_id])
                if speed >= agent.state.speed:
                    continue
                command = carla.command.ApplyVehicleControl(agent.actor, control)
                commands.append(command)
            if not commands:
                break
            self.__client.apply_batch_sync(commands)
            self.__world.tick()
        self.__warmed_up = True

    def __take_actions(self, observation: ip.Observation):
        commands = []
        for agent_id, agent in self.agents.items():
            if agent is None:
                continue

            control = agent.next_control(observation)
            if control is None:
                self.remove_agent(agent.agent_id)
                continue

            command = carla.command.ApplyVehicleControl(agent.actor, control)
            commands.append(command)
        self.__client.apply_batch_sync(commands)

    def __get_current_observation(self) -> ip.Observation:
        actor_list = self.__world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        agent_id_lookup = dict([(a.actor_id, a.agent_id) for a in self.agents.values() if a is not None])
        frame = {}
        for vehicle in vehicle_list:
            transform = vehicle.get_transform()
            heading = np.deg2rad(-transform.rotation.yaw)
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            state = ip.AgentState(time=self.__timestep,
                                  position=np.array([transform.location.x, -transform.location.y]),
                                  velocity=np.array([velocity.x, -velocity.y]),
                                  acceleration=np.array([acceleration.x, -acceleration.y]),
                                  heading=heading)
            if vehicle.id in agent_id_lookup:
                agent_id = agent_id_lookup[vehicle.id]
            else:
                agent_id = vehicle.id
            frame[agent_id] = state
        return ip.Observation(frame, self.scenario_map)

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
    def recording(self) -> bool:
        """ Whether we are recording the simulation. """
        return self.__record

    @property
    def recording_path(self) -> str:
        """ The save path of the recording. """
        return self.__record_path
