import random
import numpy as np

import carla
from carla import Transform, Location, Rotation

from igp2.agent import AgentState, ManeuverAgent
from igp2.simulator.carla_client import CarlaSim

carla_sim = CarlaSim(xodr='scenarios/maps/heckstrasse.xodr')

state = AgentState(time=0,
                   position=np.array([12.7, -8.8]),
                   velocity=np.array([0., 0.]),
                   acceleration=np.array([0., 0.]),
                   heading=0.8)

agent = ManeuverAgent(0, None)
carla_sim.add_agent(agent, state)

print('debug')

while(True):
    carla_sim.step()
    print(carla_sim.get_current_frame())