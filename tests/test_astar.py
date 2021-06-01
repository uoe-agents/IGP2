import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.agent import AgentState
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.recognition.astar import AStar

SCENARIOS = {"heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
             "round": Map.parse_from_opendrive("scenarios/maps/round.xodr"),
             "test_lane_change": Map.parse_from_opendrive("scenarios/maps/test_change_lane.xodr")}
plot_map(SCENARIOS["round"], midline=True)
plt.show()


class TestAStart:
    def test_search(self):
        scenario_map = SCENARIOS["round"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([41.30, -39.2]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-0.3),
            1: AgentState(time=0,
                          position=np.array([54.21, -50.4]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 5),
            2: AgentState(time=0,
                          position=np.array([64.72, -27.65]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-4 * np.pi / 3),
            3: AgentState(time=0,
                          position=np.array([78.78, -22.10]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 2 - np.pi / 6),
            4: AgentState(time=0,
                          position=np.array([86.13, -25.47]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=np.pi / 2),
        }

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")
            plt.text(agent.position[0], agent.position[1], agent_id, fontdict={"size": 10})

        astar = AStar()
        goal = PointGoal(Point([90.95, -10.06]), 0.1)
        search_to_goal = astar.search(0, frame, goal, scenario_map)

        plt.show()