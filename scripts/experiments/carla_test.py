import random
import numpy as np

import carla
from carla import Transform, Location, Rotation

from igp2.agent import AgentState
from igp2.planlibrary.maneuver import FollowLaneCL, ManeuverConfig, ManeuverAgent
from igp2.simulator.carla_client import CarlaSim

carla_sim = CarlaSim(xodr='scenarios/maps/heckstrasse.xodr')


configs = [
    ManeuverConfig({'type': 'follow-lane',
                    'termination_point': (1.1, -1.3)}),
    ManeuverConfig({'type': 'switch-left',
                    'termination_point': (16.2, -6.8)}),
    ManeuverConfig({'type': 'give-way',
                    'junction_road_id': 6,
                    'junction_lane_id': -1,
                    'termination_point': (31.8, -18.5)}),
    ManeuverConfig({'type': 'turn',
                    'junction_road_id': 6,
                    'junction_lane_id': -1,
                    'termination_point': (60.1, -18.5)})
]

agent_id = 0
position = np.array((-4.6, 3.4))
heading = 0.8
speed = 5
velocity = np.array([0., 0.])
acceleration = np.array([0., 0.])
state = AgentState(time=0, position=position, velocity=velocity,
                   acceleration=acceleration, heading=heading)
frame = {agent_id: state}

agent = ManeuverAgent(configs, 0, None)
carla_sim.add_agent(agent, state)

print('debug')

carla_sim.run()
