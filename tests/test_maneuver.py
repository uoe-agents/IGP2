import numpy as np
import matplotlib.pyplot as plt
import pytest

from igp2.core.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import ManeuverConfig, FollowLane, Turn, SwitchLaneLeft, GiveWay
from igp2.planlibrary.maneuver_cl import PIDController, AdaptiveCruiseControl
from igp2.opendrive.plot_map import plot_map

scenario = Map.parse_from_opendrive(f"scenarios/maps/heckstrasse.xodr")


class TestManeuver:

    def test_follow_lane(self):
        config = ManeuverConfig({'type': 'follow-lane',
                                 'initial_lane_id': 2,
                                 'final_lane_id': 2,
                                 'termination_point': (27.1, -19.8)})
        agent_id = 0
        position = np.array((8.4, -6.0))
        heading = -0.6
        speed = 10
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        acceleration = np.array([0, 0])
        agent_0_state = AgentState(time=0, position=position, velocity=velocity,
                                   acceleration=acceleration, heading=heading)
        frame = {0: agent_0_state}
        maneuver = FollowLane(config, agent_id, frame, scenario)

        plot_map(scenario, markings=True, midline=False)
        plt.plot(*position, marker="o", color="b")
        plt.plot(*list(zip(*maneuver.trajectory.path)), color="b")
        plt.show()

        assert 21 < maneuver.trajectory.length < 26
        assert 2.1 < maneuver.trajectory.duration < 2.6

        # spacing between points should be roughly 1m
        path_lengths = np.linalg.norm(np.diff(maneuver.trajectory.path, axis=0), axis=1)
        assert np.all(path_lengths < 1.1 * maneuver.POINT_SPACING)
        assert np.all(path_lengths > 0.9 * maneuver.POINT_SPACING)

        # velocities should be close to 10
        assert np.all(np.abs(maneuver.trajectory.velocity - 10) < 0.5)

        # start and end should be close to state/config
        assert np.allclose(maneuver.trajectory.path[0], (8.4, -6.0), atol=1)
        assert np.allclose(maneuver.trajectory.path[-1], (27.1, -19.8), atol=1)

    def test_turn(self):
        config = ManeuverConfig({'termination_point': (61.7, -46.3),
                                 'junction_road_id': 6, 'junction_lane_id': -1})
        agent_id = 0
        position = np.array((45.55, -20.1))
        heading = -2.8
        speed = 10
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        acceleration = np.array([0, 0])
        frame = {0: AgentState(time=0, position=position, velocity=velocity,
                               acceleration=acceleration, heading=heading)}

        maneuver = Turn(config, agent_id, frame, scenario)

        plot_map(scenario, markings=True, midline=False)
        plt.plot(*position, marker="o", color="b")
        plt.plot(*list(zip(*maneuver.trajectory.path)), color="b")
        plt.show()

        # spacing between points should be roughly 1m
        path_lengths = np.linalg.norm(np.diff(maneuver.trajectory.path, axis=0), axis=1)
        assert np.all(path_lengths < 1.1 * maneuver.POINT_SPACING)
        assert np.all(path_lengths > 0.9 * maneuver.POINT_SPACING)

        # start and end should be close to state/config
        assert np.allclose(maneuver.trajectory.path[0], (45.55, -20.1), atol=1)
        assert np.allclose(maneuver.trajectory.path[-1], (61.7, -46.3), atol=1)

    def test_switch_lane_left(self):
        agent_id = 0
        position = np.array((10, -6.8))
        heading = -0.6
        speed = 10
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        acceleration = np.array([0, 0])
        frame = {0: AgentState(0, position=position, velocity=velocity,
                               acceleration=acceleration, heading=heading)}

        config = ManeuverConfig({'type': 'switch-lane',
                                 'termination_point': (31.3, -19.2)})
        maneuver = SwitchLaneLeft(config, agent_id, frame, scenario)
        # spacing between points should be roughly 1m

        plot_map(scenario, markings=True, midline=False)
        plt.plot(*position, marker="o", color="b")
        plt.plot(*list(zip(*maneuver.trajectory.path)), color="b")
        plt.show()

        path_lengths = np.linalg.norm(np.diff(maneuver.trajectory.path, axis=0), axis=1)
        assert np.all(path_lengths < 1.1 * maneuver.POINT_SPACING)
        assert np.all(path_lengths > 0.9 * maneuver.POINT_SPACING)

        # start and end should be close to state/config
        assert np.allclose(maneuver.trajectory.path[0], (10, -6.8), atol=1)
        assert np.allclose(maneuver.trajectory.path[-1], (31.3, -19.2), atol=1)

        # each point should be with 90 degrees of overall direction
        directions = np.diff(maneuver.trajectory.path, axis=0)
        overall_direction = (config.termination_point - position).reshape((2, 1))
        assert np.all((directions @ overall_direction) > 0)

    def test_give_way_no_oncoming(self):
        config = ManeuverConfig({'termination_point': (31.7, -19.8),
                                 'junction_road_id': 6, 'junction_lane_id': -1})

        agent_id = 0
        position = np.array((10.6, -4.1))
        heading = -0.6
        speed = 10
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        acceleration = np.array([0, 0])

        frame = {0: AgentState(0, position=position, velocity=velocity,
                               acceleration=acceleration, heading=heading)}

        maneuver = GiveWay(config, agent_id, frame, scenario)

        plot_map(scenario, markings=True, midline=False)
        plt.plot(*position, marker="o", color="b")
        plt.plot(*list(zip(*maneuver.trajectory.path)), color="b")
        plt.show()

        # there should be no stops
        assert np.all(maneuver.trajectory.velocity > 1)

    def test_give_way_oncoming(self):
        config = ManeuverConfig({'termination_point': (31.7, -19.8),
                                 'junction_road_id': 5, 'junction_lane_id': -1})

        agent_id = 0
        position = np.array((10.6, -4.1))
        heading = -0.6
        speed = 10
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        acceleration = np.array([0, 0])

        speed_2 = 4
        position_2 = np.array((65.4, -41.7))
        heading_2 = 2.5
        velocity_2 = speed_2 * np.array([np.cos(heading_2), np.sin(heading_2)])

        frame = {0: AgentState(0, position=position, velocity=velocity,
                               acceleration=acceleration, heading=heading),
                 1: AgentState(0, position=position_2, velocity=velocity_2,
                               acceleration=acceleration, heading=heading_2)}

        maneuver = GiveWay(config, agent_id, frame, scenario)

        plot_map(scenario, markings=True, midline=False)
        plt.plot(*position, marker="o", color="b")
        plt.plot(*list(zip(*maneuver.trajectory.path)), color="b")
        plt.show()

        # there should be one stop
        assert np.any(maneuver.trajectory.velocity < 1)

    def test_p_controller(self):
        controller = PIDController(0.5)
        action = controller.next_action(0.5)
        assert action == 0.25

    def test_acc(self):
        controller = AdaptiveCruiseControl(5, 5, 4, 2, 1.5)
        acceleration = controller.get_acceleration(15, 12, 8, 20)
        assert acceleration == pytest.approx(-4.75, 0.1)
