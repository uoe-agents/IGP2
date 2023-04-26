import numpy as np
import matplotlib.pyplot as plt

from igp2.core.agentstate import AgentState
from igp2.core.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.recognition.astar import AStar

SCENARIOS = {
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
    "bendplatz": Map.parse_from_opendrive("scenarios/maps/bendplatz.xodr"),
    "frankenberg": Map.parse_from_opendrive("scenarios/maps/frankenberg.xodr"),
    "round": Map.parse_from_opendrive("scenarios/maps/neuweiler.xodr"), }


class TestAStar:
    def test_search_bendplatz(self):
        scenario_map = SCENARIOS["bendplatz"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([30.76, -10.31]),
                          velocity=10,
                          acceleration=0.0,
                          heading=-np.pi / 4),
            1: AgentState(time=0,
                          position=np.array([28.71, -2.10]),
                          velocity=4,
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
            4: AgentState(time=0,
                          position=np.array([38.06, -18.65]),
                          velocity=3,
                          acceleration=0.0,
                          heading=-3 * np.pi / 4),
        }

        goals = {
            0: PointGoal(np.array([37.78, -50.40]), 2.0),
            1: PointGoal(np.array([74.14, -57.66]), 2.0),
            2: PointGoal(np.array([31.29, -56.72]), 2.0),
            3: PointGoal(np.array([76.54, -60.40]), 2.0)
        }

        colors = "rgbyk"

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id % len(colors)])
            plt.text(*agent.position, agent_id)

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            for traj in trajectories:
                plt.plot(*list(zip(*traj.path)), color=colors[agent_id % len(colors)])

        plt.show()

    def test_search_round(self):
        scenario_map = SCENARIOS["round"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([41.1, -41.0]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-0.3),
            1: AgentState(time=0,
                          position=np.array([58.31, -50.6]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 3),
            2: AgentState(time=0,
                          position=np.array([79.2, -28.65]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-17 * np.pi / 18),
            3: AgentState(time=0,
                          position=np.array([147.1, -58.7]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=17 * np.pi / 18),
        }

        goals = {
            0: PointGoal(np.array([104.3, -3.8]), 2),
            1: PointGoal(np.array([13.5, -22.7]), 2),
            2: PointGoal(np.array([136.0, -66.5]), 2),
            3: PointGoal(np.array([18.54, -28.1]), 2)
        }

        colors = {0: "r", 1: "g", 2: "b", 3: "y"}

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id])
            plt.text(*agent.position, agent_id)

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            plt.plot(*goal.center, marker="x")
            for traj in trajectories:
                plt.plot(*list(zip(*traj.path)), color=colors[agent_id])

        plt.show()

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
            0: PointGoal(np.array([26.6, 1.91]), 2),
            1: PointGoal(np.array([101.6, -24.08]), 2),
            2: PointGoal(np.array([5.5, -39.27]), 2),
            3: PointGoal(np.array([52.3, -52.9]), 2)
        }

        colors = {0: "r", 1: "g", 2: "b", 3: "y"}

        plot_map(scenario_map, markings=True, midline=True)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id])
            plt.text(*agent.position, agent_id)

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            plt.plot(*goal.center, marker="x")
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
            3: AgentState(time=0,
                          position=np.array([61.35, -13.9]),
                          velocity=5.5,
                          acceleration=0.0,
                          heading=-np.pi + 0.4),
            4: AgentState(time=0,
                          position=np.array([45.262, -20.5]),
                          velocity=8.5,
                          acceleration=0.0,
                          heading=np.pi - 0.6),
            5: AgentState(time=0,
                          position=np.array([41.33, -19.56]),
                          velocity=8.5,
                          acceleration=0.0,
                          heading=np.pi),
            6: AgentState(time=0,
                          position=np.array([39.24, -25.7]),
                          velocity=8.5,
                          acceleration=0.0,
                          heading=-np.pi/3),
        }

        goals = {
            0: PointGoal(np.array([90.12, -68.061]), 2),
            1: PointGoal(np.array([61.17, -18.1]), 2),
            2: PointGoal(np.array([61.17, -18.1]), 2),
            3: PointGoal(np.array([90.12, -68.061]), 2),
            4: PointGoal(np.array([21.09, -6.4]), 2),
            5: PointGoal(np.array([21.09, -6.4]), 2),
            6: PointGoal(np.array([90.12, -68.061]), 2),
        }

        colors = {0: "r", 1: "g", 2: "b", 3: "y", 4: "r", 5: "g", 6: "b"}

        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o", color=colors[agent_id])
            plt.text(*agent.position, agent_id)

        astar = AStar()
        for agent_id in goals:
            goal = goals[agent_id]
            trajectories, actions = astar.search(agent_id, frame, goal, scenario_map)
            for traj in trajectories:
                plt.plot(*list(zip(*traj.path)), color=colors[agent_id])

        plt.show()