# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional
import logging

import numpy as np
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Polygon, LineString, Point, MultiLineString
from shapely.ops import linemerge
from dataclasses import dataclass

from igp2.opendrive.elements.geometry import cut_segment
from igp2.opendrive.elements.road_record import RoadRecord

logger = logging.getLogger(__name__)

class LaneOffset(RoadRecord):
    """The lane offset record defines a lateral shift of the lane reference line
    (which is usually identical to the road reference line).

    (Section 5.3.7.1 of OpenDRIVE 1.4)

    """


class LeftLanes:
    """ """

    sort_direction = False

    def __init__(self):
        self._lanes = []

    @property
    def lanes(self):
        """ """
        self._lanes.sort(key=lambda x: x.id, reverse=self.sort_direction)
        return self._lanes


class CenterLanes(LeftLanes):
    """ """


class RightLanes(LeftLanes):
    """ """

    sort_direction = True


class LaneWidth(RoadRecord):
    """Entry for a lane describing the width for a given position.
    (Section 5.3.7.2.1.2.0 of OpenDRIVE 1.4)


    start_offset being the offset of the entry relative to the preceding lane section record
    """

    def __init__(
            self,
            *polynomial_coefficients: float,
            idx: int = None,
            start_offset: float = None
    ):
        self.idx = idx
        self.length = 0
        super().__init__(*polynomial_coefficients, start_pos=start_offset)

        self._constant_width = None
        if self.polynomial_coefficients[0] > 0.0 and all(
                [np.isclose(v, 0.0) for v in self.polynomial_coefficients[1:]]):
            self._constant_width = self.polynomial_coefficients[0]

    @property
    def start_offset(self):
        """Return start_offset, which is the offset of the entry to the
        start of the lane section.
        """
        return self._start_pos

    @start_offset.setter
    def start_offset(self, value):
        self._start_pos = value

    @property
    def constant_width(self):
        return self._constant_width

    def width_at(self, ds: float) -> float:
        """ Return the width of the lane at ds

        Args:
            ds: Distance along the lane

        Returns:
            Distance at ds
        """
        if self._start_pos > ds or ds > self._start_pos + self.length:
            raise RuntimeError(f"Distance of {ds} is out of bounds for length {self.length} from {self._start_pos}!")

        if self._constant_width is not None:
            return self._constant_width

        return np.polyval(list(reversed(self.polynomial_coefficients)), ds - self._start_pos)


class LaneBorder(LaneWidth):
    """Describe lane by width in respect to reference path.

    (Section 5.3.7.2.1.2.0 of OpenDRIVE 1.4)

    Instead of describing lanes by their width entries and, thus,
    invariably depending on influences of inner
    lanes on outer lanes, it might be more convenient to just describe
    the outer border of each lane
    independent of any inner lanes’ parameters.
    """


@dataclass
class LaneMarker:
    """ Dataclass for storing RoadMarking data in the OpenDrive standard """
    width: float
    color: str
    weight: str
    type: str
    idx: int
    start_offset: float

    @property
    def color_to_rgb(self):
        return {
            "standard": (0, 0, 0),
            "yellow": (0.859, 0.839, 0.239)
        }[self.color]

    @property
    def type_to_linestyle(self):
        return [{
                    "none": None,
                    "solid": "-",
                    "broken": (0, (10, 10))
                }[t] for t in self.type.split(" ")]

    @property
    def plot_width(self):
        return 10 * (self.width if self.width > 0.0 else 0.13) + \
               (0.13 if self.weight == "bold" else 0)


class LaneTypes:
    NONE = "none"
    DRIVING = "driving"
    STOP = "stop"
    SHOULDER = "shoulder"
    BIKING = "biking"
    SIDEWALK = "sidewalk"
    BORDER = "border"
    RESTRICTED = "restricted"
    PARKING = "parking"
    BIDIRECTIONAL = "bidirectional"
    MEDIAN = "median"
    SPECIAL1 = "special1"
    SPECIAL2 = "special2"
    SPECIAL3 = "special3"
    ROADWORKS = "roadworks"
    TRAM = "tram"
    RAIL = "rail"
    ENTRY = "entry"
    EXIT = "exit"
    OFFRAMP = "offramp"
    ONRAMP = "onramp"

    all_types = [
        "none", "driving", "stop", "shoulder", "biking", "sidewalk", "border", "restricted", "parking",
        "bidirectional", "median", "special1", "special2", "special3", "roadWorks", "tram", "rail",
        "entry", "exit", "offRamp", "onRamp"
    ]


class Lane:
    """ Represent a single Lane of a LaneSection in the OpenDrive standard """

    def __init__(self, parent_road, lane_section):
        self._parent_road = parent_road
        self._lane_section = lane_section

        self._id = None
        self._type = None
        self._level = None
        self._link = RoadLaneLink()
        self._widths = []
        self._borders = []
        self._markers = []
        self.has_border_record = False
        self._boundary = None
        self._ref_line = None
        self._midline = None

    def __repr__(self):
        return f"Lane(id={self.id}) on Road(id={self._parent_road.id})"

    @property
    def lane_section(self) -> "LaneSection":
        """ The LaneSection this Lane is contained in """
        return self._lane_section

    @property
    def parent_road(self) -> "Road":
        """ The Road this Lane is contained in """
        return self._parent_road

    @property
    def id(self) -> int:
        """ Unique identifier of the lane within its lane section """
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = int(value)

    @property
    def type(self) -> str:
        """ Return the type of this Lane """
        return self._type

    @type.setter
    def type(self, value):
        if value not in LaneTypes.all_types:
            raise Exception(f"The specified lane type '{self._type}' is not a valid type.")
        self._type = str(value)

    @property
    def level(self):
        """ """
        return self._level

    @level.setter
    def level(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")
        self._level = value == "true"

    @property
    def link(self):
        """ """
        return self._link

    @property
    def widths(self) -> List[LaneWidth]:
        """ List of LaneWidths describing the width of the lane along its length """
        self._widths.sort(key=lambda x: x.start_offset)
        return self._widths

    @widths.setter
    def widths(self, value: List[LaneWidth]):
        self._widths = value

    @property
    def length(self):
        return self.midline.length

    @property
    def constant_width(self) -> Optional[float]:
        """ If not None, then the lane has constant width as given by this property"""
        if self._widths is not None and len(self._widths) > 0:
            if all([width.constant_width == self._widths[0].constant_width for width in self._widths]):
                return self._widths[0].constant_width
        return None

    @property
    def boundary(self) -> Polygon:
        """ The boundary Polygon of the lane """
        return self._boundary

    @property
    def reference_line(self) -> LineString:
        """ Return the reference line of the lane. For right lane this is the right edge; for left lanes the left edge;
         for center lanes it is the road midline.
        """
        return self._ref_line

    @property
    def midline(self) -> LineString:
        """ Return a line along the center of the lane"""
        return self._midline

    @property
    def borders(self) -> List[LaneBorder]:
        """ Get all LaneBorders of this Lane """
        return self._borders

    @property
    def markers(self) -> List[LaneMarker]:
        """ Get all LaneMarkers of this Lane """
        return self._markers

    def distance_at(self, point: np.ndarray) -> float:
        """ Return the distance along the Lane midline at the given point.

        Args:
            point: The point to check

        Returns:
            distance float
        """
        p = Point(point)
        return self.midline.project(p)

    def point_at(self, distance: float) -> np.ndarray:
        """ Return the point along the Lane midline at the given distance.

        Args:
            distance: The point to check

        Returns:
             1d numpy array
        """
        return np.array(self.midline.interpolate(distance))

    def calculate_boundary(self, reference_line, resolution: float = 0.5) -> Tuple[Polygon, LineString]:
        """ Calculate boundary of lane.
        Store the resulting value in boundary.

        Args:
            reference_line: The reference line of the lane. Should be the right edge for left lanes and the left edge
                for right lanes
            resolution: The spacing between samples when widths are non-constant

        Returns:
            The calculated lane boundary
        """
        assert self.parent_road is not None

        direction = np.sign(self.id)
        side = "left" if self.id > 0 else "right"
        if direction == 0:  # Ignore center-line
            buffer = Polygon()
            ref_line = reference_line

        elif self.constant_width is not None:
            # TODO: Support LaneWidth offset for constant width setting?
            buffer = reference_line.buffer(direction * (self.constant_width + 1e-5),
                                           cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre,
                                           single_sided=True)
            ref_line = reference_line.parallel_offset(self.constant_width,
                                                      side=side,
                                                      join_style=JOIN_STYLE.mitre)
            if side == "right":
                ref_line = LineString(ref_line.coords[::-1])

        else:  # Sample lane width at given resolution
            ls = []
            for ds in np.arange(0, reference_line.length, resolution):
                p_start = np.array(reference_line.interpolate(ds))
                p_end = np.array(reference_line.interpolate(min(reference_line.length, ds + 0.5)))
                if np.allclose(p_start, p_end):
                    break
                width = self.get_width_at(ds)
                if width is None:
                    logger.debug(f"LaneWidth of {self} is None at d={ds}!")
                    w = 0.01
                else:
                    w = max(0.01, width.width_at(ds))

                d = direction * np.array([[0, -1], [1, 0]]) @ (p_end - p_start)
                d /= np.linalg.norm(d)
                ls.append(tuple(p_start + w * d))

            ls.append(tuple(p_end + w * d))
            segment = list(zip(*cut_segment(reference_line, 0, reference_line.length).xy))
            buffer = Polygon(segment + ls[::-1])
            ref_line = LineString(ls)

        self._boundary = buffer
        self._ref_line = ref_line
        return buffer, ref_line

    def get_midline(self, resolution: float = 0.5) -> LineString:
        """ Calculate the midline of lane.
        Store the resulting value in linestring.

        Args:
            resolution: The spacing between samples when widths are non-constant

        Returns:
            The calculated midline
        """
        assert self.parent_road is not None
        reference_line = self.reference_line
        if self.id == 0:
            self._midline = self._ref_line
            return self._midline

        if self.constant_width is not None:
            side = "left" if self.id < 0 else "right"
            mid_line = reference_line.parallel_offset(self.constant_width / 2,
                                                      side=side,
                                                      join_style=JOIN_STYLE.round)

        else:  # Sample lane width at given resolution
            ls = []

            for ds in np.arange(0, reference_line.length, resolution):
                p_start = np.array(reference_line.interpolate(ds))
                p_end = np.array(reference_line.interpolate(min(reference_line.length, ds + 0.5)))
                if np.allclose(p_start, p_end):
                    break
                width = self.get_width_at(ds)
                if width is None:
                    logger.debug(f"LaneWidth of {self} is None at d={ds}!")
                    w = 0.01
                else:
                    w = max(0.01, width.width_at(ds)) / 2

                direction = -np.sign(self.id)
                d = direction * np.array([[0, -1], [1, 0]]) @ (p_end - p_start)
                d /= np.linalg.norm(d)
                ls.append(tuple(p_start + w * d))
            ls.append(tuple(p_end + w * d))
            mid_line = LineString(ls)

        self._midline = mid_line
        return mid_line

    def get_width_idx(self, width_idx) -> Optional[LaneWidth]:
        """ Get the LaneWidth object with the given index.

        Args:
            width_idx: The queried index

        Returns:
            LaneWidth with given index or None
        """
        for width in self._widths:
            if width.idx == width_idx:
                return width
        return None

    def get_width_at(self, ds: float) -> Optional[LaneWidth]:
        """ Calculate lane width at the given distance

        Args:
            ds: Distance along the lane

        Returns:
            Width at given distance or None if invalid distance
        """
        for width in self._widths:
            if width.start_offset <= ds < width.start_offset + width.length:
                return width
        return None

    def get_last_lane_width_idx(self):
        """ Returns the index of the last width sector of the lane """
        num_widths = len(self._widths)
        if num_widths > 1:
            return num_widths - 1
        return 0

    def get_heading_at(self, ds: float) -> float:
        """ Gets the heading at a distance along the lane

        Args:
            ds: Distance along the lane

        Returns:
            Heading at given distance

        """
        road_heading = self.parent_road.plan_view.calc_geometry(self.lane_section.start_distance + ds)[1]
        if self.id > 0:
            road_heading = road_heading % (2 * np.pi) - np.pi
        return road_heading

    def get_direction_at(self, ds: float) -> np.ndarray:
        """ Gets the direction at a position along the lane

        Args:
            ds: Distance along the lane

        Returns:
            2d vector giving direction
        """
        heading = self.get_heading_at(ds)
        return np.array([np.cos(heading), np.sin(heading)])


class RoadLaneLink:
    """ Represent a Link between two Lanes in separate LaneSections """

    def __init__(self):
        self._predecessor_id = None
        self._predecessor = None
        self._successor_id = None
        self._successor = None

    @property
    def predecessor_id(self) -> int:
        """ Lane ID of the preceding Lane"""
        return self._predecessor_id

    @predecessor_id.setter
    def predecessor_id(self, value):
        self._predecessor_id = int(value)

    @property
    def predecessor(self) -> Lane:
        """ The preceding Lane"""
        return self._predecessor

    @predecessor.setter
    def predecessor(self, value: Lane):
        self._predecessor = value

    @property
    def successor_id(self) -> int:
        """ Lane ID of the successor Lane """
        return self._successor_id

    @successor_id.setter
    def successor_id(self, value):
        self._successor_id = int(value)

    @property
    def successor(self) -> Lane:
        """ Lane ID of the successor Lane """
        return self._successor

    @successor.setter
    def successor(self, value: Lane):
        self._successor = value


class LaneSection:
    """The lane section record defines the characteristics of a road cross-section.
     (Section 5.3.7.2 of OpenDRIVE 1.4)
    """

    def __init__(self, road=None):
        self.idx = None
        self._start_ds = None
        self._single_side = None
        self._left_lanes = LeftLanes()
        self._center_lanes = CenterLanes()
        self._right_lanes = RightLanes()

        self._parent_road = road

    @property
    def single_side(self):
        """Indicator if lane section entry is valid for one side only."""
        return self._single_side

    @single_side.setter
    def single_side(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")

        self._single_side = value == "true"

    @property
    def start_distance(self):
        """ Starting distance of the LaneSection along the Road """
        return self._start_ds

    @property
    def left_lanes(self):
        """Get list of sorted lanes always starting in the middle (lane id -1)"""
        return self._left_lanes.lanes

    @property
    def center_lanes(self):
        """ The center Lane of the LaneSection """
        return self._center_lanes.lanes

    @property
    def right_lanes(self):
        """Get list of sorted lanes always starting in the middle (lane id 1)"""
        return self._right_lanes.lanes

    @property
    def all_lanes(self):
        """ Concatenate all lanes into a single array. Lanes are not sorted by id!"""
        return self._left_lanes.lanes + self._center_lanes.lanes + self._right_lanes.lanes

    def get_lane(self, lane_id: int) -> Lane:
        """ Get a Lane by its ID

        Args:
            lane_id: The ID of the Lane to look-up
        """
        for lane in self.all_lanes:
            if lane.id == lane_id:
                return lane
        return None

    @property
    def parent_road(self):
        """ The Road in which this LaneSection is contained """
        return self._parent_road


class Lanes:
    """ Collection class for LaneSections of a Road """

    def __init__(self):
        self._lane_offsets = []
        self._lane_sections = []

    @property
    def lane_offsets(self):
        """ Offsets of LaneSections """
        self._lane_offsets.sort(key=lambda x: x.start_offset)
        return self._lane_offsets

    @property
    def lane_sections(self) -> List[LaneSection]:
        """ Return all LaneSections sorted from start of the Road to its end """
        self._lane_sections.sort(key=lambda x: x._start_ds)
        return self._lane_sections

    def get_lane_section(self, lane_section_idx):
        """ Get a LaneSection by index

        Args:
            lane_section_idx: The index of the LaneSection to look-up
        """
        for laneSection in self.lane_sections:
            if laneSection.idx == lane_section_idx:
                return laneSection
        return None

    def get_last_lane_section_idx(self):
        """ Get the index of the last LaneSection in this Road """
        num_lane_sections = len(self.lane_sections)
        if num_lane_sections > 1:
            return num_lane_sections - 1
        return 0
