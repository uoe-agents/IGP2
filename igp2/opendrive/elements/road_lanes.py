# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional
import logging

import numpy as np
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Polygon, LineString, Point, MultiLineString
from shapely.ops import linemerge
from dataclasses import dataclass

from igp2.opendrive.elements.geometry import cut_segment, ramer_douglas, normalise_angle
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
        self.length = 0.0
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
        """ Return the width of the lane at ds.

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

    def calculate_boundary_and_midline(self, reference_line, resolution: float = 0.5) -> Tuple[Polygon, LineString]:
        """ Calculate boundary of lane.
        Store the resulting value in boundary.

        Args:
            reference_line: The reference line of the lane. Should be the right edge for left lanes and the left edge
                for right lanes
            resolution: The spacing between samples when widths are non-constant

        Returns:
            The calculated lane boundary
        """

        def _append_width(_ds, _width):
            parent_dist = self.lane_section.start_distance + _ds
            ref_dist = reference_line.project(self.parent_road.plan_view.midline.interpolate(parent_dist))
            point = np.array(reference_line.interpolate(ref_dist))
            w = max(0.01, _width.width_at(_ds))
            theta = normalise_angle(self.get_heading_at(_ds, False)) + direction * np.pi / 2
            normal = np.array([np.cos(theta), np.sin(theta)])
            refs.append(tuple(point + w * normal))
            mids.append(tuple(point + w / 2 * normal))

        direction = np.sign(self.id)
        side = "left" if self.id > 0 else "right"
        if direction == 0:  # Ignore center-line
            buffer = Polygon()
            ref_line = mid_line = reference_line

        else:
            refs = []
            mids = []

            start_ds = self.widths[0].start_offset
            for width_idx, width in enumerate(self.widths):
                end_ds = start_ds + width.length

                start_segment = None
                if width_idx == 0 and start_ds > 0.0:
                    start_segment = cut_segment(reference_line, 0, start_ds).parallel_offset(1e-2, side=side)
                    refs += list(start_segment.coords)
                    mids += list(start_segment.coords)

                if width.constant_width is not None:
                    end_segment = cut_segment(reference_line, start_ds,
                                              end_ds if width_idx < len(self._widths) - 1 else reference_line.length)

                    ref_line = end_segment.parallel_offset(width.constant_width, side=side,
                                                           join_style=JOIN_STYLE.mitre)

                    mid_line = end_segment.parallel_offset(width.constant_width / 2, side=side,
                                                           join_style=JOIN_STYLE.round)

                    refs += list(ref_line.coords)
                    refs = refs[::-1] if side == "right" else refs
                    mids += list(mid_line.coords)
                    mids = mids[::-1] if side == "right" else mids

                else:  # Sample lane width at given resolution
                    for ds in np.arange(start_ds, end_ds, resolution):
                        _append_width(ds, width)
                    if width_idx == len(self.widths) - 1:
                        _append_width(end_ds, width)

                start_ds = end_ds

            if side == "left":
                mids = mids[::-1]

            buffer = Polygon(list(reference_line.coords) + refs[::-1])
            ref_line = LineString(refs)
            mid_line = LineString(mids)

        self._boundary = buffer
        self._ref_line = ref_line
        self._midline = mid_line

        return buffer, ref_line

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

    def get_width_at(self, ds: float) -> float:
        """ Calculate lane width at the given distance

        Args:
            ds: Distance along the lane

        Returns:
            Width at given distance or None if invalid distance
        """
        if len(self._widths) == 1:
            return self._widths[0].width_at(ds)

        for width_idx, width in enumerate(self._widths):
            if width_idx < self.get_last_lane_width_idx():
                width = self.get_width_idx(width_idx)
                next_width = self.get_width_idx(width_idx + 1)

                if width.start_offset <= ds < next_width.start_offset:
                    return width.width_at(ds)
            else:
                return width.width_at(ds)

    def get_last_lane_width_idx(self):
        """ Returns the index of the last width sector of the lane """
        num_widths = len(self._widths)
        if num_widths > 1:
            return num_widths - 1
        return 0

    def get_heading_at(self, ds: float, lane_direction: bool = True) -> float:
        """ Gets the heading at a distance along the lane using the parent road's geometry.

        Returns the final heading of the lane if the distance is larger than length of the parent Road.

        Args:
            ds: Distance along the lane
            lane_direction: If True, then account for the direction of the lane in the heading. Else, just
                retrieve the heading of the parent road instead.

        Returns:
            Heading at given distance

        """
        try:
            road_heading = self.parent_road.plan_view.calc_geometry(self.lane_section.start_distance + ds)[1]
        except Exception as e:
            logger.debug(str(e))
            road_heading = self.parent_road.plan_view.calc_geometry(self.parent_road.plan_view.length)[1]

        if lane_direction and self.id > 0:
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

        self.length = 0.0

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