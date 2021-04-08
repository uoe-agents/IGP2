import os

from igp2.opendrive.map import Map


class TestMap:
    def test_parse(self):
        for scenario in os.listdir("scenarios"):
            Map.parse_from_opendrive(f"scenarios/{scenario}")