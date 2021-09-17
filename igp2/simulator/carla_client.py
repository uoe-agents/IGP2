import time
import numpy as np
import carla
import subprocess
import psutil
import platform
import logging

from carla import Transform, Location, Rotation, Vector3D
from igp2.agent.agent import Agent
from igp2.agent.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.vehicle import Observation

logger = logging.getLogger(__name__)

class CarlaSim:
    """ An interface to the CARLA simulator """

    MAX_ACCELERATION = 5

    def __init__(self, fps=20, xodr=None, port=2000, carla_path='/opt/carla-simulator'):
        """ Launch the CARLA simulator and define a CarlaSim object

        Args:
            fps: number of frames simulated per second
            xodr: path to a .xodr OpenDrive file defining the road layout
            port: port to use for communication with the CARLA simulator
            carla_path: path to the root directory of the CARLA simulator
        """
        self.scenario_map = Map.parse_from_opendrive(xodr)
        if platform.system() == "Windows":
            args = [f'{carla_path}/CarlaUE4.exe', '-quality-level=Low', f'-carla-rpc-port={port}']
        elif platform.system() == "Linux":
            args = [f'{carla_path}/CarlaUE4.sh', '-quality-level=Low', f'-carla-rpc-port={port}']
        self.__carla_process = subprocess.Popen(args)
        self.__port = port
        self.__client = carla.Client('localhost', port)
        self.__client.set_timeout(10.0)  # seconds
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
        self._dead_agents = []
        self.__actor_ids = {}

    def __wait_for_server(self):
        for i in range(10):
            try:
                self.__client = carla.Client('localhost', self.__port)
                self.__client.set_timeout(10.0)  # seconds
                self.__client.get_world()
                return
            except RuntimeError:
                pass
        self.__client.get_world()

    def __del__(self):
        proc = self.__carla_process
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()

    def load_opendrive_world(self, xodr):
        with open(xodr, 'r') as f:
            opendrive = f.read()
        self.__client.generate_opendrive_world(opendrive)

    def step(self):
        """ Advance the simulation by one time step"""
        self.__world.tick()
        self.__timestep += 1
        frame = self.__get_current_frame()
        observation = Observation(frame, self.scenario_map)
        self.__take_actions(observation)

    def add_agent(self, agent: Agent):
        """ Add a vehicle to the simulation

        Args:
            agent: agent to add
            state: initial state of the agent
        """
        blueprint_library = self.__world.get_blueprint_library()
        blueprint = blueprint_library.find('vehicle.audi.a2')
        state = agent.state
        yaw = np.rad2deg(state.heading)
        transform = Transform(Location(x=state.position[0], y=-state.position[1], z=0.1), Rotation(yaw=yaw))
        actor = self.__world.spawn_actor(blueprint, transform)
        actor.set_target_velocity(Vector3D(state.velocity[0], -state.velocity[1], 0.))
        self.__actor_ids[agent.agent_id] = actor.id
        self.agents[agent.agent_id] = agent

    def __get_current_frame(self):
        actor_list = self.__world.get_actors()
        frame = {}
        for agent_id in self.agents:
            if agent_id not in self._dead_agents:
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

    def __take_actions(self, observation: Observation):
        actor_list = self.__world.get_actors()

        for agent_id, agent in list(self.agents.items()):
            if agent_id not in self._dead_agents:
                actor = actor_list.find(self.__actor_ids[agent_id])
                action = agent.next_action(observation)
                if action is None or agent.done(observation):
                    actor.destroy()
                    self._dead_agents.append(agent_id)
                    continue
                control = carla.VehicleControl()
                norm_acceleration = action.acceleration/self.MAX_ACCELERATION
                if action.acceleration >= 0:
                    control.throttle = min(1., norm_acceleration)
                    control.brake = 0.
                else:
                    control.throttle = 0.
                    control.brake = min(-norm_acceleration, 1.)
                control.steer = -action.steer_angle
                control.hand_brake = False
                control.manual_gear_shift = False

                actor.apply_control(control)

    def run(self, steps=400):
        """ Run the simulation for a number of time steps """
        for i in range(steps):
            logger.debug(f"CARLA step {i} of {steps}.")
            self.step()
            time.sleep(1/self.__fps)
