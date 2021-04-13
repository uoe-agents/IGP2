import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, List, Dict
from datetime import datetime
from shapely.geometry import Point
from lxml import etree

from igp2.opendrive.elements.geometry import normalise_angle
from igp2.opendrive.elements.junction import Junction
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
        self.__date = datetime.strptime(self.__opendrive.header.date, "%c")
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

        junctions = {}
        for junction in self.__opendrive.junctions:
            junction.calculate_boundary()

            assert junction.id not in junctions
            junctions[junction.id] = junction
        self.__junctions = junctions

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
            if road.boundary is not None and road.boundary.contains(point):
                candidates.append(road)
        return candidates

    def lanes_at(self, point: Union[Point, Tuple[float, float], np.ndarray], drivable: bool = True) -> List[Lane]:
        """ Return all lanes passing through the given point

        Args:
            point: Point in cartesian coordinates
            drivable: If True, only return drivable lanes

        Returns:
            A list of all viable lanes or empty list
        """
        candidates = []
        point = Point(point)
        roads = self.roads_at(point)
        for road in roads:
            for lane_section in road.lanes.lane_sections:
                lanes = lane_section.drivable_lanes if drivable else \
                    lane_section.left_lanes + lane_section.right_lanes
                for lane in lanes:
                    if lane.boundary is not None and lane.boundary.contains(point):
                        candidates.append(lane)
        return candidates

    def best_road_at(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float) -> Road:
        """ Get the road at the given point with the closest direction to the heading

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians

        Returns:
            A Road passing through point with its direction closest to the given heading
        """
        point = Point(point)
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

        warn_threshold = np.pi / 18
        if best_diff > warn_threshold:  # Warning if angle difference was too large
            logger.warning(f"Best angle difference of {np.rad2deg(best_diff)} > "
                           f"{np.rad2deg(warn_threshold)} on road {best}!")
        return best

    def best_lane_at(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float) -> Lane:
        """ Get the lane at the given point with the closest direction to the heading

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians

        Returns:
            A Lane passing through point with its direction closest to the given heading
        """
        point = Point(point)
        road = self.best_road_at(point, heading)
        for lane_section in road.lanes.lane_sections:
            for lane in lane_section.all_lanes:
                if lane.boundary is not None and lane.boundary.contains(point):
                    return lane
        return None

    def junction_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> Junction:
        """ Get the Junction at a given point

        Args:
            point: Location to check in cartesian coordinates

        Returns:
            A Junction object or None
        """
        point = Point(point)
        for junction_id, junction in self.junctions.items():
            if junction.boundary is not None and junction.boundary.contains(point):
                return junction
        return None

    def adjacent_lanes_at(self, point: Union[Point, Tuple[float, float], np.ndarray],
                          heading: float = None, same_direction: bool = False) -> List[Lane]:
        """ Return all adjacent lanes on the same Road

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians
            same_direction: If True, only return lanes in the same direction as the current Lane

        Returns:
            A list of all adjacent Lane objects on the same Road
        """
        raise NotImplementedError()

    def next_element(self, point: Union[Point, Tuple[float, float], np.ndarray],
                     heading: float = None) -> Union[Road, Junction]:
        """ Get the next element of the current Road in the direction of the heading

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians

        Returns:
            A Road or a Junction
        """
        raise NotImplementedError()

    def get_legal_turns(self, point, heading: float = None) -> List[Road]:
        """ Get all legal turns (as Roads) from a given point in the given heading.
        If the point falls within a junction, return the connecting Road it is on.

        Args:
            point: A point in cartesian coordinates
            heading: Heading in radians

        Returns:
            A list of Roads that are legal to enter from the given point. Or a single element list containing the
                current Road if the point falls in the junction. Or an empty list
        """
        raise NotImplementedError()

    def plot(self, midline: bool = True, road_ids: bool = True, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """ Draw the road layout of the map

        Args:
            midline: True if the midline of roads should be drawn
            road_ids: If True, then the IDs of roads will be drawn
            ax: Axes to draw on

        Keyword Args:
            road_color: Plot color of the road boundary (default: black)
            midline_color: Color of the midline

        Returns:
            The axes onto which the road layout was drawn
        """
        colors = plt.get_cmap("tab10").colors

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

            color = kwargs.get("midline_color",
                               colors[road_id % len(colors)] if road_ids else "r")
            if midline:
                ax.plot(road.midline.xy[0], road.midline.xy[1], color=color)

            if road_ids:
                mid_point = len(road.midline.xy) // 2
                ax.text(road.midline.xy[0][mid_point], road.midline.xy[1][mid_point], road.id,
                        color=color, fontsize=15)

        return ax

    def is_valid(self):
        """ Checks if the Map geometry is valid. """
        for road in self.roads.values():
            if road.boundary is None or not road.boundary.is_valid:
                return False

            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.left_lanes + lane_section.right_lanes:
                    if lane.boundary is None or not lane.boundary.is_valid:
                        return False

        for junction in self.junctions.values():
            if junction.boundary is None or not junction.boundary.is_valid:
                return False

        return True

    @property
    def name(self) -> str:
        """ Name for the map """
        return self.__name

    @property
    def date(self) -> datetime:
        """ Date when the map was created """
        return self.__date

    @property
    def geo_reference(self) -> str:
        """ Geo-reference parameters for geo-location """
        return self.__geo_reference

    @property
    def roads(self) -> Dict[int, Road]:
        """ Dictionary of all roads in the map with keys the road IDs """
        return self.__roads

    @property
    def junctions(self):
        return self.__junctions

    @property
    def north(self) -> float:
        """ North boundary of the map"""
        return self.__north

    @property
    def south(self) -> float:
        """ South boundary of the map"""
        return self.__south

    @property
    def east(self) -> float:
        """ East boundary of the map"""
        return self.__east

    @property
    def west(self) -> float:
        """ West boundary of the map"""
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
    map = Map.parse_from_opendrive("scenarios/heckstrasse.xodr")
    map.is_valid()
    map.plot()
    plt.show()
