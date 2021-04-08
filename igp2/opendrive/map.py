import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Union, Tuple, List
from datetime import datetime
from shapely.geometry import Point
from lxml import etree

from igp2.opendrive.elements.geometry import normalise_angle
from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road import Road
from igp2.opendrive.elements.roadLanes import Lane
from igp2.opendrive.parser import parse_opendrive

logger = logging.getLogger()


class Map(object):
    """ Define a map object based on the OpenDrive standard """

    def __init__(self, opendrive: OpenDrive = None):
        """ Create a map object given the parsed OpenDrive file

        Args:
            opendrive: A class describing the parsed contents of the OpenDrive file
        """
        self.__opendrive = opendrive

        self.__process_header()
        self.__process_road_layout()

    def __process_header(self):
        self.__name = self.__opendrive.header.name
        # self.__date = datetime.strptime(self.__opendrive.header.date, "%c")
        self.__north = float(self.__opendrive.header.north)
        self.__west = float(self.__opendrive.header.west)
        self.__south = float(self.__opendrive.header.south)
        self.__east = float(self.__opendrive.header.east)
        self.__geo_reference = self.__opendrive.header.geo_reference

    def __process_road_layout(self):
        roads = {}

        for road in self.__opendrive.roads:
            road.plan_view.precalculate(linestring=True)
            road.calculate_boundary()

            assert road.id not in roads
            roads[road.id] = road

        self.__roads = roads

    def roads_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> List[Road]:
        """ Find all roads that pass through the given point

        Args:
            point: Point in cartesian coordinates

        Returns:
            A list of all viable roads or empty list
        """
        point = Point(point)
        candidates = []
        for id, road in self.roads.items():
            if road.boundary.contains(point):
                candidates.append(road)
        return candidates

    def lanes_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> List[Lane]:
        """ Return all lanes passing through the given point

        Args:
            point: Point in cartesian coordinates

        Returns:
            A list of all viable lanes or empty list
        """
        candidates = []
        roads = self.roads_at(point)
        for road in roads:
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.drivable_lanes:
                    if lane.boundary.contains(point):
                        candidates.append(lane)
        return candidates

    def best_road_at(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float) -> Road:
        """ Get the road at the given point with the closest direction as the heading

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians

        Returns:
            A Road passing through point with its direction closest to the given heading
        """
        roads = self.roads_at(point)
        assert len(roads) > 0
        if len(roads) == 1:
            return roads[0]

        best = None
        best_diff = np.inf
        heading = normalise_angle(heading)
        for road in roads:
            _, angle = road.plan_view.calc(road.midline.project(point))
            if road.junction is None and np.abs(heading - angle) > np.pi / 2:
                heading *= -1
            diff = np.abs(heading - angle)
            if diff < best_diff:
                best = road
                best_diff = diff

        thresh = np.pi / 18
        if best_diff > thresh:  # Warning if angle difference was too large
            logger.warning(f"Best angle difference of {np.rad2deg(best_diff)} > {np.rad2deg(thresh)} on road {best}!")
        return best

    def best_lane_at(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float) -> Lane:
        pass

    def plot(self, midline: bool = True, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """ Draw the road layout of the map

        Args:
            ax: Axes to draw on
            midline: True if the midline of roads should be drawn

        Keyword Args:
            road_color: Plot color of the road boundary (default: black)
            midline_color: Color of the midline (default: red)

        Returns:
            The axes onto which the road layout was drawn
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.set_xlim([self.west, self.east])
        ax.set_ylim([self.south, self.north])

        for road_id, road in self.roads.items():
            boundary = road.boundary.boundary
            if boundary.geom_type == "LineString":
                ax.plot(boundary.xy[0], boundary.xy[1],
                        color=kwargs.get("road_color", "k"))
            elif boundary.geom_type == "MultiLineString":
                for b in boundary:
                    ax.plot(b.xy[0], b.xy[1],
                            color=kwargs.get("road_color", "orange"))

            if midline:
                ax.plot(road.midline.xy[0], road.midline.xy[1],
                        color=kwargs.get("midline_color", "r"))

        return ax

    @property
    def name(self):
        return self.__name

    @property
    def date(self):
        return self.__date

    @property
    def geo_reference(self):
        return self.__geo_reference

    @property
    def roads(self):
        return self.__roads

    @property
    def north(self):
        return self.__north

    @property
    def south(self):
        return self.__south

    @property
    def east(self):
        return self.__east

    @property
    def west(self):
        return self.__west

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
    map = Map.parse_from_opendrive("scenarios/bendplatz.xodr")
    map.plot()
    plt.show()

    # p = Point(48.5, -32.6)
    p = Point(50, -44)
    h = 1.9
    rs = map.roads_at(p)
    rsb = map.best_road_at(p, h)
    ls = map.lanes_at(p)
