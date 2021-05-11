import os
import numpy as np

from igp2.agent import AgentState
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import ManeuverConfig, FollowLane


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

        assert 21 < maneuver.trajectory.length < 26
        assert 2.1 < maneuver.trajectory.duration < 2.6

        # spacing between points should be rougly 1m
        path_lengths = np.linalg.norm(np.diff(maneuver.trajectory.path, axis=0), axis=1)
        assert np.all(path_lengths < 1.1)
        assert np.all(path_lengths > 0.9)

        # velocities should be close to 10
        assert np.all(np.abs(maneuver.trajectory.velocity - 10) < 0.5)

        # start and end should be close to state/config
        assert np.allclose(maneuver.trajectory.path[0], (8.4, -6.0), atol=1)
        assert np.allclose(maneuver.trajectory.path[-1], (27.1, -19.8), atol=1)

