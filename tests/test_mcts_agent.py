import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.agent.agentstate import AgentMetadata, AgentState
from igp2.agent.mcts_agent import MCTSAgent
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.vehicle import Observation

scenarios = {"town1": Map.parse_from_opendrive("scenarios/maps/Town01.xodr")}
basic_meta = AgentMetadata(**AgentMetadata.CAR_DEFAULT)
frame = {
    0: AgentState(0,
                  np.array([88.38, -38.62]),
                  np.array([0.0, -5.0]),
                  np.array([0.0, 0.0]),
                  -np.pi / 2,
                  basic_meta),
    1: AgentState(0,
                  np.array([334.58, 1.22]),
                  np.array([1.0, -5.0]),
                  np.array([0.0, 0.0]),
                  -np.pi / 4,
                  basic_meta)
}


class TestMCTSAgent:
    def test_get_goals(self):
        agent_id = 1
        scenario_map = scenarios["town1"]
        agent = MCTSAgent(agent_id,
                          frame[agent_id],
                          1,
                          basic_meta,
                          scenario_map,
                          PointGoal(np.array([186.23, -2.03]), 2.0),
                          view_radius=5)
        goals = agent.get_goals(Observation(frame, scenario_map))

        plot_map(scenario_map, markings=True)
        plt.plot(*frame[agent_id].position, marker="o")
        view_circle = Point(*frame[agent_id].position).buffer(agent.view_radius).boundary
        plt.plot(*list(zip(*view_circle.coords)))
        for goal in goals:
            plt.plot(*goal.center, marker="x", color="b")
        plt.show()