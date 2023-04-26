import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.core.agentstate import AgentMetadata, AgentState
from igp2.agents.mcts_agent import MCTSAgent
from igp2.core.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.core.vehicle import Observation

scenarios = {
    "town1": Map.parse_from_opendrive("scenarios/maps/Town01.xodr"),
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr")
}


class TestMCTSAgent:
    def test_get_goals_heckstrasse(self):
        frame = {
            0: AgentState(time=0,
                          position=np.array([6.0, 0.7]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-0.6),
        }

        agent_id = 0
        scenario_map = scenarios["heckstrasse"]
        agent = MCTSAgent(agent_id,
                          frame[agent_id],
                          1,
                          scenario_map,
                          PointGoal(np.array([186.23, -2.03]), 2.0),
                          view_radius=100)
        agent.get_goals(Observation(frame, scenario_map))

        plot_map(scenario_map, markings=True)
        plt.plot(*frame[agent_id].position, marker="o")
        view_circle = Point(*frame[agent_id].position).buffer(agent.view_radius).boundary
        plt.plot(*list(zip(*view_circle.coords)))
        for goal in agent._goals:
            plt.plot(*goal.center, marker="x", color="b")
        plt.show()

    def test_get_goals_town01(self):
        basic_meta = AgentMetadata(**AgentMetadata.CAR_DEFAULT)
        frame = {
            0: AgentState(0,
                          np.array([88.38, -38.62]),
                          np.array([0.0, -5.0]),
                          np.array([0.0, 0.0]),
                          -np.pi / 2),
            1: AgentState(0,
                          np.array([334.58, 1.22]),
                          np.array([1.0, -5.0]),
                          np.array([0.0, 0.0]),
                          -np.pi / 4)
        }

        agent_id = 1
        scenario_map = scenarios["town1"]
        agent = MCTSAgent(agent_id,
                          frame[agent_id],
                          1,
                          scenario_map,
                          PointGoal(np.array([186.23, -2.03]), 2.0),
                          view_radius=5)
        agent.get_goals(Observation(frame, scenario_map))

        plot_map(scenario_map, markings=True)
        plt.plot(*frame[agent_id].position, marker="o")
        view_circle = Point(*frame[agent_id].position).buffer(agent.view_radius).boundary
        plt.plot(*list(zip(*view_circle.coords)))
        for goal in agent._goals:
            plt.plot(*goal.center, marker="x", color="b")
        plt.show()
