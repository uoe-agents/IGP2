import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.agent import AgentState
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.recognition.astar import AStar

SCENARIOS = {
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
    "bendplatz": Map.parse_from_opendrive("scenarios/maps/bendplatz.xodr"),
    "frankenberg": Map.parse_from_opendrive("scenarios/maps/frankenberg.xodr"),
    "round": Map.parse_from_opendrive("scenarios/maps/round.xodr"), }


class TestAStar:
    def test_search_round(self):
        pass

    def test_search_frankenberg(self):
        scenario_map = SCENARIOS["frankenberg"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([32.0, -36.54]),
                          velocity=8,
                          acceleration=0.0,
                          heading=np.pi / 12),
            1: AgentState(time=0,
                          position=np.array([64.1, -70.13]),
                          velocity=10,
                          acceleration=0.0,
                          heading=10 * np.pi / 18),
            2: AgentState(time=0,
                          position=np.array([81.3, -23.5]),
                          velocity=10,
                          acceleration=0.0,
                          heading=-17 * np.pi / 18),
            3: AgentState(time=0,
                          position=np.array([21.6, 1.23]),
                          velocity=4,
                          acceleration=0.0,
                          heading=-np.pi / 3),
        }

        goals = {
            0: PointGoal(np.array([26.6, 1.91]), 1.0),
            1: PointGoal(np.array([101.6, -24.08]), 1.0),
            2: PointGoal(np.array([5.5, -39.27]), 1.0),
            3: PointGoal(np.array([52.3, -52.9]), 1.0)
        }

        colors = {0: "r", 1: "g", 2: "b", 3: "y"}

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id])

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            for traj in trajectories:
                plt.plot(*list(zip(*traj.path)), color=colors[agent_id])

        plt.show()

    def test_search_bendplatz(self):
        scenario_map = SCENARIOS["bendplatz"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([30.76, -10.31]),
                          velocity=8,
                          acceleration=0.0,
                          heading=-np.pi / 4),
            1: AgentState(time=0,
                          position=np.array([28.71, -2.10]),
                          velocity=10,
                          acceleration=0.0,
                          heading=-np.pi / 4),
            2: AgentState(time=0,
                          position=np.array([74.58, -49.29]),
                          velocity=10,
                          acceleration=0.0,
                          heading=3 * np.pi / 4),
            3: AgentState(time=0,
                          position=np.array([78.55, -4.67]),
                          velocity=4,
                          acceleration=0.0,
                          heading=-3 * np.pi / 4),
        }

        goals = {
            0: PointGoal(np.array([37.78, -50.40]), 1.0),
            1: PointGoal(np.array([74.14, -57.66]), 1.0),
            2: PointGoal(np.array([31.29, -56.72]), 1.0),
            3: PointGoal(np.array([76.54, -60.40]), 1.0)
        }

        colors = {0: "r", 1: "g", 2: "b", 3: "y"}

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id])

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            for traj in trajectories:
                plt.plot(*list(zip(*traj.path)), color=colors[agent_id])

        plt.show()

    def test_search_heckstrasse(self):
        scenario_map = SCENARIOS["heckstrasse"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([6.0, 0.7]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-0.6),
            1: AgentState(time=0,
                          position=np.array([19.7, -13.5]),
                          velocity=8.5,
                          acceleration=0.0,
                          heading=-0.6),
            2: AgentState(time=0,
                          position=np.array([73.2, -47.1]),
                          velocity=11.5,
                          acceleration=0.0,
                          heading=np.pi - 0.6),
        }

        goals = {
            0: PointGoal(np.array([91.2, -67.7]), 0.5),
            1: PointGoal(np.array([75.6, -15.2]), 0.5),
            2: PointGoal(np.array([76.8, -14.1]), 0.5),
        }

        colors = {0: "r", 1: "g", 2: "b"}

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id])

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            plt.plot(*list(zip(*trajectories[0].path)), color=colors[agent_id])

        plt.show()
