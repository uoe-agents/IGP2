import numpy as np
import matplotlib.pyplot as plt

from igp2.core.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.macro_action import ChangeLaneLeft, ChangeLaneRight, Exit, MacroActionFactory, Continue

SCENARIOS = {"heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
             "bendplatz": Map.parse_from_opendrive("scenarios/maps/bendplatz.xodr"),
             "round": Map.parse_from_opendrive("scenarios/maps/neuweiler.xodr"),
             "test_lane_change": Map.parse_from_opendrive("scenarios/maps/test_change_lane.xodr")}


class TestMacroAction:
    def test_lane_change_bendplatz(self):
        scenario_map = SCENARIOS["bendplatz"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([29.0, -2.3]),
                          velocity=5.5,
                          acceleration=0.0,
                          heading=-np.pi / 4),
            1: AgentState(time=0,
                          position=np.array([31.1, -11.0]),
                          velocity=5.5,
                          acceleration=0.0,
                          heading=-np.pi / 4),
            2: AgentState(time=0,
                          position=np.array([41.6, -21.4]),
                          velocity=5.5,
                          acceleration=0.0,
                          heading=-np.pi / 8),
            3: AgentState(time=0,
                          position=np.array([68.0, -46.6]),
                          velocity=4.5,
                          acceleration=0.0,
                          heading=3 * np.pi / 4),
        }

        plot_map(scenario_map, markings=True)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")

        lane_change = ChangeLaneRight(0, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = ChangeLaneLeft(1, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = Continue(2, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = ChangeLaneRight(3, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        plt.show()

    def test_applicability(self):
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

        applicables = {
            0: [Exit, ChangeLaneLeft],
            1: [Exit, ChangeLaneLeft],
            2: [ChangeLaneRight, ContinueNextExit],
            3: [ChangeLaneRight, Exit],
            4: [Exit]
        }
        for agent_id, state in frame.items():
            actions = MacroActionFactory.get_applicable_actions(state, scenario_map)
            assert all([a in actions for a in applicables[agent_id]])

    def test_lane_change_test_map(self):
        scenario_map = SCENARIOS["test_lane_change"]
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
            # 2: AgentState(time=0,  # This agent will fail for now since the road splits
            #               position=np.array([71.7, 1.27]),
            #               velocity=4.5,
            #               acceleration=0.0,
            #               heading=np.pi),
            3: AgentState(time=0,
                          position=np.array([111.0, -1.34]),
                          velocity=9.5,
                          acceleration=0.0,
                          heading=np.pi / 8.5),
            4: AgentState(time=0,
                          position=np.array([128.7, -0.49]),
                          velocity=4.5,
                          acceleration=0.0,
                          heading=np.pi / 6),
            5: AgentState(time=0,
                          position=np.array([137.0, 8.5]),
                          velocity=10.0,
                          acceleration=0.0,
                          heading=np.pi / 4),
        }
        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")

        lane_change = ChangeLaneLeft(0, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="b")

        lane_change = ChangeLaneRight(1, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        # lane_change = ChangeLaneRight(2, frame, scenario_map, True)
        # trajectory = lane_change.get_trajectory().path
        # plt.plot(trajectory[:, 0], trajectory[:, 1], color="green")

        lane_change = ChangeLaneRight(3, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="red")

        lane_change = ChangeLaneLeft(4, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="purple")

        lane_change = ChangeLaneRight(5, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="brown")

        plt.show()

    def test_turn_round(self):
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

        turn = Exit(np.array([53.44, -47.522]), 0, frame, scenario_map, True)
        trajectory = turn.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")
        turn.final_frame[0].position += 0.15 * np.array([np.cos(turn.final_frame[0].heading),
                                                         np.sin(turn.final_frame[0].heading)])

        lane_change = ChangeLaneLeft(0, turn.final_frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")

        continue_next_exit = ContinueNextExit(0, lane_change.final_frame, scenario_map, True)
        trajectory = continue_next_exit.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")
        continue_next_exit.final_frame[0].position += 0.15 * np.array([np.cos(continue_next_exit.final_frame[0].heading),
                                                                       np.sin(continue_next_exit.final_frame[0].heading)])

        lane_change = ChangeLaneRight(0, continue_next_exit.final_frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")

        for agent_id, agent in lane_change.final_frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")

        turn = Exit(np.array([95.77, -52.74]), 0, lane_change.final_frame, scenario_map, True)
        trajectory = turn.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")


        plt.show()

    def test_lane_change_heckstrasse(self):
        scenario_map = SCENARIOS["heckstrasse"]
        frame = {
            0: AgentState(time=0,
                          position=np.array([26.9, -19.3]),
                          velocity=5.5,
                          acceleration=0.0,
                          heading=-np.pi / 8),
            1: AgentState(time=0,
                          position=np.array([6.0, 0.7]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 8),
            2: AgentState(time=0,
                          position=np.array([22.89, -12.9]),
                          velocity=2,
                          acceleration=0.0,
                          heading=-np.pi / 8),
            3: AgentState(time=0,
                          position=np.array([29.9, -21.9]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 8),
        }

        plot_map(scenario_map, markings=True)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")

        foo = ChangeLaneLeft.applicable(frame[0], scenario_map)
        lane_change = ChangeLaneLeft(0, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = ChangeLaneRight(1, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = ChangeLaneRight(2, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        # lane_change = ChangeLaneLeft(3, frame, scenario_map, True)
        # trajectory = lane_change.get_trajectory().path
        # plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        plt.show()

    def test_turn_heckstrasse(self):
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
        plot_map(scenario_map, markings=True, midline=False)
        for agent_id, agent in frame.items():
            plt.plot(agent.position[0], agent.position[1], marker="o")

        lane_change = Exit(np.array([49.44, -23.1]), 0, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")

        lane_change = Exit(np.array([62.34, -46.67]), 1, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="orange")

        lane_change = Exit(np.array([63.2, -18.5]), 2, frame, scenario_map, True)
        trajectory = lane_change.get_trajectory().path
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="green")

        plt.show()
