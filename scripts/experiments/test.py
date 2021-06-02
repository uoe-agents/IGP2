import matplotlib.pyplot as plt
import logging

from igp2 import setup_logging
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.agent import AgentState
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.recognition.astar import AStar

logger = logging.getLogger(__name__)
setup_logging()

SCENARIOS = {
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
    "round": Map.parse_from_opendrive("scenarios/maps/round.xodr"),
    }

scenario_map = SCENARIOS["heckstrasse"]
frame = {
    0: AgentState(time=0,
                  position=np.array([6.0, 0.7]),
                  velocity=1.5,
                  acceleration=0.0,
                  heading=np.pi),
    1: AgentState(time=0,
                  position=np.array([19.7, -13.5]),
                  velocity=8.5,
                  acceleration=0.0,
                  heading=np.pi),
    2: AgentState(time=0,
                  position=np.array([73.2, -47.1]),
                  velocity=11.5,
                  acceleration=0.0,
                  heading=np.pi),
}

goals = {
    # 0: PointGoal(np.array([91.2, -67.7]), 0.5),
    1: PointGoal(np.array([75.6, -15.2]), 0.5),
    2: PointGoal(np.array([76.8, -13.7]), 0.5),
}
plot_map(scenario_map, markings=True, midline=False)
for agent_id, agent in frame.items():
    plt.plot(agent.position[0], agent.position[1], marker="o")


astar = AStar()
for agent_id in goals:
    goal = goals[agent_id]
    trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
    print(actions)

plt.show()
