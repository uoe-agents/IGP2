import pytest
import numpy as np
import matplotlib.pyplot as plt

from igp2.agent import AgentState
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.macro_action import ChangeLaneLeft, ChangeLaneRight, Continue


class TestMacroAction:
    def test_lane_left(self):
        scenario_map = Map.parse_from_opendrive("scenarios/maps/test_change_lane.xodr")
        frame = {
            0: AgentState(time=0,
                          position=np.array([89.9, 4.64]),
                          velocity=11.5,
                          acceleration=0.0,
                          heading=np.pi),
            1: AgentState(time=0,
                          position=np.array([79.7, 1.27]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=np.pi),
            2: AgentState(time=0,
                          position=np.array([71.7, 1.27]),
                          velocity=4.5,
                          acceleration=0.0,
                          heading=np.pi),
            3: AgentState(time=0,
                          position=np.array([111.0, -1.34]),
                          velocity=9.5,
                          acceleration=0.0,
                          heading=np.pi / 8),
            4: AgentState(time=0,
                          position=np.array([128.7, -0.49]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=np.pi / 6),
        }
        plot_map(scenario_map, markings=True, midline=True)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")

        lane_change = ChangeLaneLeft(0, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="b")

        lane_change = ChangeLaneRight(1, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = Continue(2, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="green")

        lane_change = ChangeLaneRight(3, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="green")

        lane_change = ChangeLaneLeft(4, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="purple")

        plt.show()


