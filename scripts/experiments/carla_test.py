import random
import numpy as np

import carla
from carla import Transform, Location, Rotation

from igp2.agent import AgentState, ManeuverAgent
from igp2.planlibrary.maneuver import FollowLaneCL, ManeuverConfig
from igp2.simulator.carla_client import CarlaSim

carla_sim = CarlaSim(xodr='scenarios/maps/heckstrasse.xodr')

state = AgentState(time=0,
                   position=np.array([12.7, -8.8]),
                   velocity=np.array([0., 0.]),
                   acceleration=np.array([0., 0.]),
                   heading=0.8)

maneuver_config = ManeuverConfig({'type': 'follow-lane',
                                 'initial_lane_id': 2,
                                 'final_lane_id': 2,
                                 'termination_point': (27.1, -19.8)})

agent_id = 0
position = np.array((8.4, -6.0))
heading = -0.6
speed = 10
velocity = speed * np.array([np.cos(heading), np.sin(heading)])
acceleration = np.array([0, 0])
agent_0_state = AgentState(time=0, position=position, velocity=velocity,
                           acceleration=acceleration, heading=heading)
frame = {0: agent_0_state}

maneuver = FollowLaneCL(maneuver_config, agent_id, frame, carla_sim.scenario_map)
agent = ManeuverAgent(maneuver, 0, None)
carla_sim.add_agent(agent, state)

print('debug')

carla_sim.run()
