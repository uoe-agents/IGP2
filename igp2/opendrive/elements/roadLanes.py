# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Polygon, Point, LineString

from igp2.opendrive.elements.geometry import cut_segment
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
        if self.polynomial_coefficients[0] > 0.0 and all([np.isclose(v, 0.0) for v in self.polynomial_coefficients[1:]]):
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
        if self.start_pos > ds or ds > self.start_pos + self.length:
            raise RuntimeError(f"Distance of {ds} is out of bounds for length {self.length} from {self.start_pos}!")

        if self._constant_width is not None:
            return self._constant_width

        return np.polyval(list(reversed(self.polynomial_coefficients)), ds - self.start_pos)


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
        """ The Road this lane is contained in """
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
        if value not in self.laneTypes:
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
    def constant_width(self) -> float:
        """ If not None, then the lane has constant width as given by this property"""
        if self._widths is not None and len(self._widths) > 0:
            if all([width.constant_width == self._widths[0].constant_width for width in self._widths]):
                return self._widths[0].constant_width
        return None

    @property
    def boundary(self) -> Polygon:
        """ The boundary Polygon of the lane """
        return self._boundary

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
            buffer = reference_line.buffer(direction * (self.constant_width + 1e-5),
                                           cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre,
                                           single_sided=True)
            ref_line = reference_line.parallel_offset(self.constant_width,
                                                      side=side,
                                                      join_style=JOIN_STYLE.round)
            if side == "right":
                ref_line = LineString(ref_line.coords[::-1])
        else:  # Sample lane width at resolution
            ls = []
            max_length = min(reference_line.length, sum([w.length for w in self._widths]))
            for ds in np.arange(0, max_length, resolution):
                p_start = np.array(reference_line.interpolate(ds))
                p_end = np.array(reference_line.interpolate(min(max_length, ds + 0.5)))
                width = self.get_width_at(ds)
                if width is None:
                    raise RuntimeError(f"Width of lane is None!")
                w = max(0.01, width.width_at(ds))

                d = direction * np.array([[0, -1], [1, 0]]) @ (p_end - p_start)
                d /= np.linalg.norm(d)
                ls.append(tuple(p_start + w * d))
            ls.append(tuple(p_end + w * d))
            segment = list(zip(*cut_segment(reference_line, 0, max_length).xy))
            buffer = Polygon(segment + ls[::-1])
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
            if width.start_pos <= ds < width.start_pos + width.length:
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
        return self._start_ds

    @property
    def left_lanes(self):
        """Get list of sorted lanes always starting in the middle (lane id -1)"""
        return self._left_lanes.lanes

    @property
    def center_lanes(self):
        """ """
        return self._center_lanes.lanes

    @property
    def right_lanes(self):
        """Get list of sorted lanes always starting in the middle (lane id 1)"""
        return self._right_lanes.lanes

    @property
    def all_lanes(self):
        """Attention! lanes are not sorted by id"""
        return self._left_lanes.lanes + self._center_lanes.lanes + self._right_lanes.lanes

    @property
    def drivable_lanes(self):
        return [lane for lane in self._left_lanes.lanes if lane.type == "driving"] + \
               [lane for lane in self._right_lanes.lanes if lane.type == "driving"]

    def get_lane(self, lane_id: int) -> Lane:
        for lane in self.all_lanes:
            if lane.id == lane_id:
                return lane
        return None

    @property
    def parent_road(self):
        return self._parent_road


class Lanes:
    """ """

    def __init__(self):
        self._lane_offsets = []
        self._lane_sections = []

    @property
    def lane_offsets(self):
        self._lane_offsets.sort(key=lambda x: x.start_pos)
        return self._lane_offsets

    @property
    def lane_sections(self) -> List[LaneSection]:
        self._lane_sections.sort(key=lambda x: x._start_ds)
        return self._lane_sections

    def get_lane_section(self, lane_section_idx):
        for laneSection in self.lane_sections:
            if laneSection.idx == lane_section_idx:
                return laneSection
        return None

    def get_last_lane_section_idx(self):
        num_lane_sections = len(self.lane_sections)
        if num_lane_sections > 1:
            return num_lane_sections - 1
        return 0
