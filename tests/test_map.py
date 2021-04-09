import os
import pytest

from igp2.opendrive.map import Map


class TestMap:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.scenarios = {}
        for scenario in os.listdir("scenarios"):
            scenario = Map.parse_from_opendrive(f"scenarios/{scenario}")
            assert scenario.is_valid()
            self.scenarios[scenario.name] = scenario

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
                ret = [r.id for r in self.scenarios[name].roads_at(p)]
                assert ret == ids, p

    @pytest.mark.parametrize("drivable", [True, False])
    def test_lanes_at(self, drivable):
        tests = {
            "heckstrasse": {(16.7, -3.1): [-1],
                            (25.9, -19.4): [2],
                            (65.8, -45.4): [] if drivable else [-1],
                            (50.4, -36.7): [-1, -1],
                            (46.5, -25.7): [-1, -1, -1]},
            "test": {(1867.2, 689.98): [-2],
                     (-503.9, 2317.2): [] if drivable else [-3],
                     (-497.6, 2322.0): [1]}
        }
        for name, points in tests.items():
            for p, ids in points.items():
                ret = [l.id for l in self.scenarios[name].lanes_at(p, drivable=drivable)]
                assert ret == ids, p

    def test_best_road_at(self):
        tests = {
            "heckstrasse": {((16.7, -3.1), 2.35): 1,
                            ((25.9, -19.4), -0.78): 1,
                            ((65.8, -45.4), -0.8): 2,
                            ((47.28, -33.6), -1.22): 6,
                            ((46.5, -25.7), 2.35): 8,
                            ((46.5, -25.7), 0.39): 5,
                            ((46.5, -25.7), -1.57): 6},
            "frankenberg": {((50.71, -22.59), 1.57): 8,
                            ((50.71, -22.59), 1.8): 6,
                            ((50.71, -22.59), 2.2): 12
                            },
        }
        for name, points in tests.items():
            for (p, h), i in points.items():
                assert self.scenarios[name].best_road_at(p, h).id == i, (p, h)

    def test_best_lane_at(self):
        pass
