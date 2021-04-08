# -*- coding: utf-8 -*-
from typing import List

import numpy as np
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Polygon, Point, LineString

from igp2.opendrive.elements.road_record import RoadRecord


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
        if self.polynomial_coefficients[0] > 0.0 and all([v == 0.0 for v in self.polynomial_coefficients[1:]]):
            self._constant_width = self.polynomial_coefficients[0]

    @property
    def start_offset(self):
        """Return start_offset, which is the offset of the entry to the
        start of the lane section.
        """
        return self.start_pos

    @start_offset.setter
    def start_offset(self, value):
        self.start_pos = value

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
        if self.start_pos > ds or ds > self.length:
            raise RuntimeError(f"Distance of {ds} is out of bounds for length {self.length}!")

        if self._constant_width is not None:
            return self._constant_width

        return np.polyval(list(reversed(self.polynomial_coefficients)), ds)


class LaneBorder(LaneWidth):
    """Describe lane by width in respect to reference path.

    (Section 5.3.7.2.1.2.0 of OpenDRIVE 1.4)

    Instead of describing lanes by their width entries and, thus,
    invariably depending on influences of inner
    lanes on outer lanes, it might be more convenient to just describe
    the outer border of each lane
    independent of any inner lanes’ parameters.
    """


class Lane:
    """ """

    laneTypes = [
        "none",
        "driving",
        "stop",
        "shoulder",
        "biking",
        "sidewalk",
        "border",
        "restricted",
        "parking",
        "bidirectional",
        "median",
        "special1",
        "special2",
        "special3",
        "roadWorks",
        "tram",
        "rail",
        "entry",
        "exit",
        "offRamp",
        "onRamp",
    ]

    def __init__(self, parent_road, lane_section):
        self._parent_road = parent_road
        self.lane_section = lane_section

        self._id = None
        self._type = None
        self._level = None
        self._link = LaneLink()
        self._widths = []
        self._borders = []
        self.has_border_record = False
        self._boundary = None

    @property
    def parent_road(self):
        """ """
        return self._parent_road

    @property
    def id(self):
        """ """
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def type(self):
        """ """
        return self._type

    @type.setter
    def type(self, value):
        if value not in self.laneTypes:
            raise Exception()

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
    def widths(self):
        """ """
        self._widths.sort(key=lambda x: x.start_offset)
        return self._widths

    @widths.setter
    def widths(self, value):
        self._widths = value

    @property
    def constant_width(self) -> float:
        if len(self._widths) == 1:
            return self._widths[0].constant_width

    @property
    def boundary(self):
        return self._boundary

    def calculate_boundary(self, reference_line):
        """ Calculate boundary of lane.
        Store the resulting value in boundary.

        Args:
            reference_line: The reference line of the lane. Should be the right edge for left lanes and the left edge
                for right lanes

        Returns:
            The calculated lane boundary
        """
        assert self.parent_road is not None

        direction = np.sign(self.id)
        side = "left" if self.id > 0 else "right"
        if direction == 0:
            buffer = Polygon()
            ref_line = reference_line
        elif self.constant_width is not None:
            buffer = reference_line.buffer(direction * self.constant_width,
                                           cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre,
                                           single_sided=True)
            ref_line = reference_line.parallel_offset(self.constant_width - 1e-5,
                                                      side=side,
                                                      join_style=JOIN_STYLE.mitre)
            if side == "right":
                ref_line = LineString(ref_line.coords[::-1])
        else:
            ls = []
            for p_start, p_end in zip(reference_line.coords[:-1], reference_line.coords[1:]):
                p_start = np.array(p_start)
                p_end = np.array(p_end)
                ds = reference_line.project(Point(p_start))
                width = self.get_width_at(ds)
                if width is None:
                    raise RuntimeError(f"Width of lane is None!")
                w = width.width_at(ds) - 1e-5

                d = direction * np.array([[0, -1], [1, 0]]) @ (p_end - p_start)
                d /= np.linalg.norm(d)
                ls.append(tuple(p_start + w * d))
            ls.append(tuple(p_end + w * d))
            buffer = Polygon(list(zip(*reference_line.xy)) + ls[::-1])
            ref_line = LineString(ls)

        self._boundary = buffer
        return buffer, ref_line

    def get_width(self, width_idx) -> LaneWidth:
        for width in self._widths:
            if width.idx == width_idx:
                return width
        return None

    def get_width_at(self, ds) -> LaneWidth:
        for width in self._widths:
            if width.start_pos <= ds < width.length:
                return width
        return None

    def get_last_lane_width_idx(self):
        """Returns the index of the last width sector of the lane"""

        num_widths = len(self._widths)

        if num_widths > 1:
            return num_widths - 1

        return 0

    @property
    def borders(self):
        """ """
        return self._borders


class LaneLink:
    """ """

    def __init__(self):
        self._predecessor = None
        self._successor = None

    @property
    def predecessor_id(self):
        """ """
        return self._predecessor

    @predecessor_id.setter
    def predecessor_id(self, value):
        self._predecessor = int(value)

    @property
    def successor_id(self):
        """ """
        return self._successor

    @successor_id.setter
    def successor_id(self, value):
        self._successor = int(value)


class LaneSection:
    """The lane section record defines the characteristics of a road cross-section.

    (Section 5.3.7.2 of OpenDRIVE 1.4)

    """

    def __init__(self, road=None):
        self.idx = None
        self.sPos = None
        self._singleSide = None
        self._leftLanes = LeftLanes()
        self._centerLanes = CenterLanes()
        self._rightLanes = RightLanes()

        self._parentRoad = road

    @property
    def single_side(self):
        """Indicator if lane section entry is valid for one side only."""
        return self._singleSide

    @single_side.setter
    def single_side(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")

        self._singleSide = value == "true"

    @property
    def left_lanes(self):
        """Get list of sorted lanes always starting in the middle (lane id -1)"""
        return self._leftLanes.lanes

    @property
    def center_lanes(self):
        """ """
        return self._centerLanes.lanes

    @property
    def right_lanes(self):
        """Get list of sorted lanes always starting in the middle (lane id 1)"""
        return self._rightLanes.lanes

    @property
    def all_lanes(self):
        """Attention! lanes are not sorted by id"""
        return self._leftLanes.lanes + self._centerLanes.lanes + self._rightLanes.lanes

    @property
    def drivable_lanes(self):
        return [lane for lane in self._leftLanes.lanes if lane.type == "driving"] + \
               [lane for lane in self._rightLanes.lanes if lane.type == "driving"]

    def get_lane(self, lane_id: int) -> Lane:
        """

        Args:
          lane_id:

        Returns:

        """
        for lane in self.all_lanes:
            if lane.id == lane_id:
                return lane

        return None

    @property
    def parent_road(self):
        """ """
        return self._parentRoad


class Lanes:
    """ """

    def __init__(self):
        self._laneOffsets = []
        self._lane_sections = []

    @property
    def lane_offsets(self):
        """ """
        self._laneOffsets.sort(key=lambda x: x.start_pos)
        return self._laneOffsets

    @property
    def lane_sections(self) -> List[LaneSection]:
        """ """
        self._lane_sections.sort(key=lambda x: x.sPos)
        return self._lane_sections

    def get_lane_section(self, lane_section_idx):
        """

        Args:
          lane_section_idx:

        Returns:

        """
        for laneSection in self.lane_sections:
            if laneSection.idx == lane_section_idx:
                return laneSection

        return None

    def get_last_lane_section_idx(self):
        """ """

        num_lane_sections = len(self.lane_sections)

        if num_lane_sections > 1:
            return num_lane_sections - 1

        return 0
