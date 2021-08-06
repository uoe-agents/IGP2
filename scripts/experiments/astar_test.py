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

#Script to test astar trajectory generation

logger = logging.getLogger(__name__)
setup_logging()

SCENARIOS = {
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
    "round": Map.parse_from_opendrive("scenarios/maps/round.xodr"),
    }

scenario_map = SCENARIOS["round"]
frame = {
    0: AgentState(time=0,
                  position=np.array([96.8, -0.2]),
                  velocity=4,
                  acceleration=0.0,
                  heading=-2 * np.pi / 3),
    1: AgentState(time=0,
                  position=np.array([25.0, -36.54]),
                  velocity=4,
                  acceleration=0.0,
                  heading=-0.3),
    2: AgentState(time=0,
                  position=np.array([133.75, -61.67]),
                  velocity=4,
                  acceleration=0.0,
                  heading=5*np.pi/6),
    3: AgentState(time=0,
                  position=np.array([102.75, -48.31]),
                  velocity=4,
                  acceleration=0.0,
                  heading=np.pi/2),
}
colors = "rgbyk"

goals = {
    0: PointGoal(np.array([103.99, -5.91]), 2),
    1: PointGoal(np.array([103.99, -5.91]), 2),
    2: PointGoal(np.array([60.75, -83.77]), 2),
    3: PointGoal(np.array([60.75, -83.77]), 2),
}
plot_map(scenario_map, markings=True, midline=False)
for agent_id, agent in frame.items():
    plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id % len(colors)])
    plt.text(*agent.position, agent_id)

astar = AStar()
for agent_id in goals:
    goal = goals[agent_id]
    plt.plot(*goal.center, marker="x", color=colors[agent_id % len(colors)])
    trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
    for traj in trajectories:
        plt.plot(*list(zip(*traj.path)), color=colors[agent_id % len(colors)])

plt.show()