import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, List, Dict, Optional
from datetime import datetime
from shapely.geometry import Point

from lxml import etree

from igp2.opendrive.elements.geometry import normalise_angle
from igp2.opendrive.elements.junction import Junction, JunctionGroup
from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road import Road
from igp2.opendrive.elements.road_lanes import Lane, LaneTypes
from igp2.opendrive.parser import parse_opendrive

logger = logging.getLogger(__name__)


class Map(object):
    """ Define a map object based on the OpenDrive standard """
    PRECISION_ERROR = 1e-8  # The maximum precision error allowed when checking if two geometries contain each other

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
            road.calculate_boundary_and_midline()

            assert road.id not in roads
            roads[road.id] = road
        self.__roads = roads

        junctions = {}
        for junction in self.__opendrive.junctions:
            junction.calculate_boundary()

            assert junction.id not in junctions
            junctions[junction.id] = junction
        self.__junctions = junctions

        junction_groups = {}
        for junction_group in self.__opendrive.junction_groups:
            assert junction_group.id not in junction_groups
            junction_groups[junction_group.id] = junction_group
        self.__junction_groups = junction_groups

    def roads_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> List[Road]:
        """ Find all roads that pass through the given point

        Args:
            point: Point in cartesian coordinates

        Returns:
            A list of all viable roads or empty list
        """
        point = Point(point)
        candidates = []
        for road_id, road in self.roads.items():
            if road.boundary is not None and road.boundary.distance(point) < Map.PRECISION_ERROR:
                candidates.append(road)
        return candidates

    def lanes_at(self, point: Union[Point, Tuple[float, float], np.ndarray], drivable_only: bool = False) -> List[Lane]:
        """ Return all lanes passing through the given point

        Args:
            point: Point in cartesian coordinates
            drivable_only: If True, only return drivable lanes

        Returns:
            A list of all viable lanes or empty list
        """
        candidates = []
        point = Point(point)
        roads = self.roads_at(point)
        for road in roads:
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.boundary is not None and lane.boundary.distance(point) < Map.PRECISION_ERROR and \
                            (not drivable_only or lane.type == LaneTypes.DRIVING):
                        candidates.append(lane)
        return candidates

    def best_road_at(self, point: Union[Point, Tuple[float, float], np.ndarray],
                     heading: float = None) -> Optional[Road]:
        """ Get the road at the given point with the closest direction to heading. If no heading is given, then select
        the first viable road.

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians

        Returns:
            A Road passing through point with its direction closest to the given heading, or None.

        """
        point = Point(point)
        roads = self.roads_at(point)
        if len(roads) == 0:
            logger.warning(f"No roads found at point: {point}!")
            return None
        if len(roads) == 1 or heading is None:
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
            logger.debug(f"Best angle difference of {np.rad2deg(best_diff)} > "
                         f"{np.rad2deg(warn_threshold)} at {point} on road {best}!")
        return best

    def best_lane_at(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float = None,
                     drivable_only: bool = False) -> Optional[Lane]:
        """ Get the lane at the given point whose direction is closest to heading

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians
            drivable_only: If True, only return a Lane if it is drivable

        Returns:
            A Lane passing through point with its direction closest to the given heading, or None.
        """
        point = Point(point)
        road = self.best_road_at(point, heading)
        if road is None:
            return None

        for lane_section in road.lanes.lane_sections:
            for lane in lane_section.all_lanes:
                if lane.boundary is not None and lane.boundary.distance(point) < Map.PRECISION_ERROR and \
                        lane.id != 0 and (not drivable_only or lane.type == LaneTypes.DRIVING):
                    return lane
        return None

    def junction_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> Optional[Junction]:
        """ Get the Junction at a given point

        Args:
            point: Location to check in cartesian coordinates

        Returns:
            A Junction object or None
        """
        point = Point(point)
        for junction_id, junction in self.junctions.items():
            if junction.boundary is not None and junction.boundary.distance(point) < Map.PRECISION_ERROR:
                return junction
        return None

    def adjacent_lanes_at(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float = None,
                          same_direction: bool = False, drivable_only: bool = False) -> List[Lane]:
        """ Return all adjacent lanes on the same Road

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians
            same_direction: If True, only return lanes in the same direction as the current Lane
            drivable_only: If True, only return a Lane if it is drivable

        Returns:
            A list of all adjacent Lane objects on the same Road
        """
        point = Point(point)
        current_lane = self.best_lane_at(point, heading)
        current_section = current_lane.lane_section
        direction = np.sign(current_lane.id)

        adjacents = []
        for lane in current_section.all_lanes:
            if lane.id != current_lane.id and lane.id != 0:
                dirs_equal = np.sign(lane.id) == direction
                drivable = lane.type == LaneTypes.DRIVING
                if same_direction and drivable_only:
                    if dirs_equal and drivable:
                        adjacents.append(lane)
                elif same_direction:
                    if dirs_equal:
                        adjacents.append(lane)
                elif drivable_only:
                    if drivable:
                        adjacents.append(lane)
                else:
                    adjacents.append(lane)
        return adjacents

    def in_roundabout(self, point: Union[Point, Tuple[float, float], np.ndarray], heading: float) -> bool:
        """ Determines whether the vehicle is currently in a roundabout. A roundabout is either a connector road in a
        junction with a junction group of type 'roundabout', or a road whose predecessor and successor are both in the
        same roundabout junction.

        Args:
            point: Point in cartesian coordinates
            heading: Heading in radians

        Returns:
            True if the vehicle is one a road in a roundabout.
        """
        road = self.best_road_at(point, heading)
        if road is None:
            raise ValueError(f"No road found at {point}.")
        return self.road_in_roundabout(road)

    def road_in_roundabout(self, road: Road) -> bool:
        """ Calculate whether a road is in a roundabout. A roundabout road is either a connector road in a
        junction with a junction group of type 'roundabout', or a road whose predecessor and successor are both in the
        same roundabout junction.

        Args:
            road: The Road to check

        Returns:
            True if the road is part of a roundabout
        """
        junction = road.junction
        predecessor = road.link.predecessor
        successor = road.link.successor

        # Dead-end roads cannot be in roundabouts
        if predecessor is None or successor is None:
            return False

        predecessor = predecessor.element
        successor = successor.element
        if junction is not None:
            if junction.junction_group is not None and junction.junction_group.type == "roundabout":
                # Handle all combinations of links while in a roundabout junction
                if isinstance(predecessor, Road) and isinstance(successor, Road):
                    return self.road_in_roundabout(predecessor) and self.road_in_roundabout(successor)
                elif isinstance(predecessor, Junction) and isinstance(successor, Road):
                    return predecessor.junction_group is not None and \
                           predecessor.junction_group == junction.junction_group \
                           and self.road_in_roundabout(successor)
                elif isinstance(successor, Junction) and isinstance(predecessor, Road):
                    return successor.junction_group is not None and \
                           successor.junction_group == junction.junction_group \
                           and self.road_in_roundabout(predecessor)
                else:
                    return predecessor.junction_group is not None and \
                           predecessor.junction_group == junction.junction_group and \
                           successor.junction_group is not None and \
                           successor.junction_group == junction.junction_group
            else:
                return False

        if not isinstance(predecessor, Junction) or not isinstance(successor, Junction):
            return False

        return (predecessor.junction_group == successor.junction_group is not None and
                predecessor.junction_group.type == successor.junction_group.type == "roundabout")

    def get_lane(self, road_id: int, lane_id: int) -> Lane:
        """ Get a certain lane given the road id and lane id from the first lane section.

        Args:
            road_id: Road ID of the road containing the lane
            lane_id: Lane ID of lane to look up

        Returns:
            Lane
        """
        return self.roads.get(road_id).lanes.lane_sections[0].get_lane(lane_id)

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
    def junctions(self) -> Dict[int, Junction]:
        return self.__junctions

    @property
    def junction_groups(self) -> Dict[int, JunctionGroup]:
        return self.__junction_groups

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
