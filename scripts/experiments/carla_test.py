import random
import numpy as np

import carla
from carla import Transform, Location, Rotation

from igp2.agent import MacroAgent
from igp2.agentstate import AgentState, AgentMetadata
from igp2.simulator.carla_client import CarlaSim

carla_sim = CarlaSim(xodr='scenarios/maps/heckstrasse.xodr')

state = AgentState(time=0,
                   position=np.array([12.7, -8.8]),
                   velocity=np.array([0., 0.]),
                   acceleration=np.array([0., 0.]),
                   heading=0.8)

agent = MacroAgent(0, state, AgentMetadata(agent_id=0,
                                           length=4.5, width=1.6,
                                           agent_type="car",
                                           initial_time=0, final_time=1), None)
carla_sim.add_agent(agent, state)

print('debug')

while(True):
    carla_sim.step()
    print(carla_sim.get_current_frame())