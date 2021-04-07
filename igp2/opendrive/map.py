import numpy as np

from typing import Union, Tuple, List
from datetime import datetime
from shapely.geometry import Point
from lxml import etree

from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road import Road
from igp2.opendrive.elements.roadLanes import Lane
from igp2.opendrive.parser import parse_opendrive


class Map(object):
    """ Define a map object based on the OpenDrive standard """
    def __init__(self, opendrive: OpenDrive = None):
        """ Create a map object given the parsed OpenDrive file

        Args:
            opendrive: A class describing the parsed contents of the OpenDrive file
        """
        self.opendrive = opendrive

        self.__process_header()
        self.__process_road_layout()

    def __process_header(self):
        self.name = self.opendrive.header.name
        self.date = datetime.strptime(self.opendrive.header.date, "%c")
        self.geo_reference = self.opendrive.header.geo_reference

    def __process_road_layout(self):
        roads = {}

        for road in self.opendrive.roads:
            road.plan_view.precalculate(linestring=True)
            road.calculate_boundary()

            assert road.id not in roads
            roads[road.id] = road

        self.roads = roads

    def roads_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> List[Road]:
        """ Find all roads that pass through the given point

        Args:
            point: Point in cartesian coordinates

        Returns:
            A list of all viable roads or None
        """
        point = Point(point)
        candidates = []
        for id, road in self.roads.items():
            if road.boundary.contains(point):
                candidates.append(road)
        return road

    def lanes_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> List[Lane]:
        pass

    def best_lane_at(self, point: Union[Point, Tuple[float, float], np.ndarray], angle: float) -> Lane:
        pass

    @classmethod
    def parse_from_opendrive(cls, file_path: str):
        """ Parse the OpenDrive file and create a new Map instance

        Args:
            file_path: The absolute/relative path to the OpenDrive file

        Returns:
            A new instance of the Map class
        """
        tree = etree.parse(file_path)
        odr = parse_opendrive(tree.getroot())
        return cls(odr)


if __name__ == '__main__':
    map = Map.parse_from_opendrive("scenarios/frankenberg.xodr")
    roads = map.roads_at((66.1, -16.25))