import os
import pytest

from igp2.opendrive.map import Map


scenarios = {}
for scenario in os.listdir("scenarios/maps"):
    scenario = Map.parse_from_opendrive(f"scenarios/maps/{scenario}")
    # assert scenario.is_valid(), scenario
    scenarios[scenario.name] = scenario


class TestMap:
    def test_roads_at(self):
        # Mapping from scenarios to points to road ids
        tests = {
            "heckstrasse": {(16.7, -3.1): [1],
                            (25.9, -19.4): [1],
                            (65.8, -45.4): [2],
                            (50.4, -36.7): [6, 7],
                            (46.5, -25.7): [5, 6, 8]},
            "frankenberg": {(48.64, -39.54): [4],
                            (49.73, -20.75): [6, 8, 12]},
            "test": {(1867.2, 689.98): [5],
                     (-237, 2487.6): [29],
                     (-50.7, 2455.9): [1, 7]}
        }
        for name, points in tests.items():
            for p, ids in points.items():
                ret = [r.id for r in scenarios[name].roads_at(p)]
                assert ret == ids, p

    @pytest.mark.parametrize("drivable_only", [True, False])
    def test_lanes_at(self, drivable_only):
        tests = {
            "heckstrasse": {(16.7, -3.1): [-1],
                            (25.9, -19.4): [2],
                            (65.8, -45.4): [] if drivable_only else [-1],
                            (50.4, -36.7): [-1, -1],
                            (46.5, -25.7): [-1, -1, -1]},
            "test": {(1867.2, 689.98): [-2],
                     (-503.9, 2317.2): [] if drivable_only else [-3],
                     (-497.6, 2322.0): [1]}
        }
        for name, points in tests.items():
            for p, ids in points.items():
                ret = [l.id for l in scenarios[name].lanes_at(p, drivable_only=drivable_only)]
                assert ret == ids, p

    def test_best_road_at(self):
        tests = {
            "heckstrasse": {((16.7, -3.1), 2.35): 1,
                            ((25.9, -19.4), -0.78): 1,
                            ((65.8, -45.4), -0.8): 2,
                            ((47.28, -33.6), -1.22): 6,
                            ((46.5, -25.7), 2.35): 8,
                            ((46.5, -25.7), 0.39): 5,
                            ((46.5, -25.7), -1.57): 6,
                            ((43.7, -28.0), -0.8): None},
            "frankenberg": {((50.71, -22.59), 1.57): 8,
                            ((50.71, -22.59), 1.8): 6,
                            ((50.71, -22.59), 2.2): 12,
                            ((0.0, 0.0), 0.0): None
                            },
        }
        for name, points in tests.items():
            for (p, h), i in points.items():
                road = scenarios[name].best_road_at(p, h)
                if road is None:
                    assert i is None, (p, h)
                else:
                    assert road.id == i, (p, h)

    @pytest.mark.parametrize("drivable_only", [True, False])
    def test_best_lane_at(self, drivable_only):
        tests = {
            "heckstrasse": {((16.7, -3.1), 2.35): -1,
                            ((25.9, -19.4), -0.8): 2,
                            ((65.8, -45.4), -0.8): -1 if not drivable_only else None,
                            ((47.28, -33.6), -1.22): -1,
                            ((46.5, -25.7), 2.35): -1,
                            ((46.5, -25.7), 0.39): -1,
                            ((46.5, -25.7), -1.57): -1,
                            ((28.8, -40.7), 0.0): None},
            "bendplatz": {((42.74, -14.13), 2.0): -1,
                          ((32.48, -5.68), 2.0): 1,
                          ((43.52, -24.49), 2.0): 2,
                          ((82.32, -67.20), -0.8): -1,
                          ((70.71, -50.21), -0.8): 1,
                          ((68.68, -40.67), -0.8): 2,
                          ((0.0, 0.0), 0.0): None
                          },
        }
        for name, points in tests.items():
            for (p, h), i in points.items():
                lane = scenarios[name].best_lane_at(p, h, drivable_only=drivable_only)
                if lane is None:
                    assert i is None
                else:
                    assert lane.id == i, (p, h)

    def test_junction_at(self):
        tests = {
            "heckstrasse": {(46.3, -25.4): 0,
                            (56.2, -37.8): None,
                            (16.7, -3.1): None},
            "bendplatz": {(55.91, -34.49): 0},
            "frankenberg": {(47.3, -33.2): 0},
            "test": {(-395.2, 2197.8): 26,
                     (-202.5, 2373.0): 27,
                     (-52.2, 2458.7): 0}
        }
        for name, points in tests.items():
            for p, i in points.items():
                junction = scenarios[name].junction_at(p)
                if junction is None:
                    assert i is None, p
                else:
                    assert junction.id == i, p

    @pytest.mark.parametrize("same_dir", [True, False])
    @pytest.mark.parametrize("drivable_only", [True, False])
    def test_adjacent_lanes_at(self, same_dir, drivable_only):
        tests = {
            "heckstrasse": {((16.7, -3.1), 2.35): [] if same_dir else [1, 2],
                            ((25.9, -19.4), -0.8): [1] if same_dir else [1, -1],
                            ((71.3, -49.1), -0.8): [-2] if same_dir else [1, -2],
                            ((91.1, -69.1), -0.8): [] if same_dir and drivable_only else
                                                   [-1] if same_dir else
                                                   [1] if drivable_only else
                                                   [1, -1]},
            "test": {((-196.27, 2540.6), 0.78): [] if same_dir and drivable_only else
                                                [-2, -3] if same_dir else
                                                [1] if drivable_only else
                                                [1, 2, 3, -2, -3],
                     ((-696.74, 2185.27), 2.0): [1] if same_dir and drivable_only else
                                                [1, 3] if same_dir else
                                                [1, -1] if drivable_only else
                                                [1, 3, -1, -2, -3],
                     },
        }
        for name, points in tests.items():
            for (p, h), ids in points.items():
                lanes = scenarios[name].adjacent_lanes_at(p, h, same_dir, drivable_only)
                ret = [l.id for l in lanes]
                assert ret == ids, (p, h, same_dir, drivable_only)

    def test_get_legal_turns(self):
        assert False
