import numpy as np

from igp2.agents.agentstate import AgentState, AgentMetadata
from igp2.planlibrary.maneuver import ManeuverConfig
from igp2.agents.maneuver_agent import ManeuverAgent
from igp2.carla.carla_client import CarlaSim

carla_sim = CarlaSim(xodr='scenarios/maps/heckstrasse.xodr')

configs = [
    ManeuverConfig({'type': 'give-way',
                    'junction_road_id': 5,
                    'junction_lane_id': -1,
                    'termination_point': (31.8, -18.5)}),
    ManeuverConfig({'type': 'turn',
                    'junction_road_id': 5,
                    'junction_lane_id': -1,
                    'termination_point': (60.1, -18.5)})
]

agent_id = 0
position = np.array((3.9, 1.3))
heading = 0.8
speed = 10
velocity = speed * np.array([np.cos(-heading), np.sin(-heading)])
acceleration = np.array([0., 0.])
state_0 = AgentState(time=0, position=position, velocity=velocity,
                     acceleration=acceleration, heading=heading)


configs1 = [
    ManeuverConfig({'type': 'follow-lane',
                    'termination_point': (66.0, -42.0)}),
    ManeuverConfig({'type': 'turn',
                    'junction_road_id': 8,
                    'junction_lane_id': -1,
                    'termination_point': (34.1, -17.2)}),
    ManeuverConfig({'type': 'follow-lane',
                    'termination_point': (-0.8, 8.4)}),
]


state_1 = AgentState(time=0, position=np.array((87.4, -56.5)), velocity=np.array([0., 0.]),
                     acceleration=np.array([0., 0.]), heading=-2.4)

frame = {agent_id: state_0, 1: state_1}
agents_metadata = AgentMetadata.default_meta_frame(frame)
fps = 20

agent = ManeuverAgent(configs, 0, state_0, fps=fps, agent_metadata=agents_metadata[0])
carla_sim.add_agent(agent)

agent1 = ManeuverAgent(configs1, 1, state_1, fps=fps, agent_metadata=agents_metadata[1])
carla_sim.add_agent(agent1)

print('debug')

carla_sim.run()
