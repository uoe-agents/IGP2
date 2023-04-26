# -*- coding: utf-8 -*-
import logging
from typing import Union, Tuple

import numpy as np

from shapely.geometry import JOIN_STYLE, Point, LineString
from shapely.ops import unary_union, substring
from shapely.geometry.polygon import Polygon

from igp2.opendrive.elements.geometry import normalise_angle, ramer_douglas
from igp2.opendrive.elements.road_plan_view import PlanView
from igp2.opendrive.elements.road_link import RoadLink
from igp2.opendrive.elements.road_lanes import Lanes
from igp2.opendrive.elements.road_elevation_profile import (
    ElevationProfile,
)
from igp2.opendrive.elements.road_lateral_profile import LateralProfile
from igp2.opendrive.elements.junction import Junction

logger = logging.getLogger(__name__)


class Road:
    """ Road object of the OpenDrive standard
    (OpenDrive 1.6.1 - Section 8)
    """

    def __init__(self):
        self._id = None
        self._name = None
        self._junction = None
        self._length = None
        self._boundary = None

        self._header = None  # TODO
        self._link = RoadLink()
        self._types = []
        self._planView = PlanView()
        self._elevation_profile = ElevationProfile()
        self._lateral_profile = LateralProfile()
        self._lanes = Lanes()

    def __eq__(self, other):
        return other.__class__ is self.__class__ and self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.name} from {self.plan_view.start_position} with length {self.plan_view.length}"

    @property
    def id(self) -> int:
        """ Unique ID of the Road """
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def name(self) -> str:
        """ Name of the Road"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def junction(self):
        """ Junction object if the Road is part of a junction """
        return self._junction

    @junction.setter
    def junction(self, value):
        if not isinstance(value, (Junction, int)) and value is not None:
            raise TypeError("Property must be a Junction or NoneType")
        if value == -1:
            value = None
        self._junction = value

    @property
    def link(self):
        """ """
        return self._link

    @property
    def types(self):
        """ """
        return self._types

    @property
    def plan_view(self) -> PlanView:
        """ PlanView describing the RoadGeometry of the Road in the OpenDrive standard
        (OpenDrive 1.6.1 - Section 7)
        """
        return self._planView

    @property
    def length(self):
        return self._planView.length

    @property
    def midline(self) -> LineString:
        """ The Road midline """
        return self.plan_view.midline

    def distance_at(self, point: Union[Point, Tuple[float, float], np.ndarray]) -> float:
        """ Return the distance along the Road midline at the given point.

        Args:
            point: The point to check

        Returns:
            distance float
        """
        p = Point(point)
        return self._planView.midline.project(p)

    def point_at(self, distance: float) -> np.ndarray:
        """ Return the point along the Road midline at the given distance.

        Args:
            distance: The point to check

        Returns:
             1d numpy array
        """
        return self._planView.calc(distance)[0]

    def calculate_road_geometry(self, resolution: float = 0.25, fix_eps: float = 1e-2):
        """ Calculate the boundary Polygon of the road.
        Calculates boundaries of lanes as a sub-function.

        Args:
            resolution: Sampling resolution for geometries
            fix_eps: If positive, then the algorithm attempts to fix sliver geometry in the map with this threshold
        """
        if self.lanes is None or self.lanes.lane_sections == []:
            return

        self.calculate_center_lane(resolution)

        boundary = Polygon()
        for ls in self.lanes.lane_sections:
            if ls.length < resolution:
                logger.debug(f"Road {self.id} skipping too short lane-section {ls.idx}. This will likely cause errors "
                             f"downstream. Consider removing the lane-section from the XODR file.")
                continue
            start_segment = ls.center_lanes[0].reference_line
            sample_distances = np.linspace(0.0, ls.length, int(ls.length / resolution) + 1)

            previous_direction = None
            reference_segment = start_segment
            reference_widths = np.zeros_like(sample_distances)
            for lane in ls.all_lanes:
                current_direction = np.sign(lane.id)
                if previous_direction is None or previous_direction != current_direction:
                    reference_segment = start_segment
                    reference_widths = np.zeros_like(sample_distances)

                lane_boundary, reference_segment, segment_widths = \
                    lane.sample_geometry(sample_distances, start_segment, reference_segment, reference_widths)

                boundary = unary_union([boundary, lane_boundary])
                previous_direction = current_direction
                reference_widths += segment_widths

        if fix_eps > 0.0:
            boundary = boundary.buffer(fix_eps, 1, join_style=JOIN_STYLE.mitre) \
                .buffer(-fix_eps, 1, join_style=JOIN_STYLE.mitre)

        if not boundary.boundary.geom_type == "LineString":
            logger.warning(f"Boundary of road ID {self.id} is not a closed a loop!")

        self._boundary = boundary

    def calculate_center_lane(self, resolution: float):
        """ Calculate center lane of the road by applying lane offsets
         and store them in the respective center lanes. """
        ref_line = self.plan_view.midline
        if not self.lanes.lane_offsets:
            center_lane = ref_line
        else:
            sample_distances = np.linspace(0.0, ref_line.length, int(ref_line.length / resolution) + 1)
            offsets = []

            for offset_idx, offset in enumerate(self.lanes.lane_offsets):
                if offset_idx == len(self.lanes.lane_offsets) - 1:
                    section_distances = sample_distances[offset.start_offset <= sample_distances]
                else:
                    next_offset = self.lanes.lane_offsets[offset_idx + 1]
                    indices = ((offset.start_offset <= sample_distances) & (
                                sample_distances < next_offset.start_offset)).nonzero()[0]
                    section_distances = sample_distances[indices]
                coefficients = list(reversed(offset.polynomial_coefficients))
                section_offset = np.polyval(coefficients, section_distances - offset.start_offset)
                offsets.append(section_offset)

            offsets = np.hstack(offsets)
            points = []
            for i, d in enumerate(sample_distances):
                point = np.array(ref_line.interpolate(d).coords[0])
                theta = normalise_angle(self.plan_view.calc(d)[1] + np.pi / 2)
                normal = np.array([np.cos(theta), np.sin(theta)])
                points.append(tuple(point + offsets[i] * normal))

            center_lane = LineString(ramer_douglas(points, dist=0.01))
            if not center_lane.is_simple:
                coords_list = []
                for non_intersecting_ls in unary_union(center_lane).geoms:
                    if non_intersecting_ls.length > 0.5:
                        coords_list.extend(non_intersecting_ls.coords)
                center_lane = LineString(coords_list)

        # Assign midlines for each center lane for every lane section
        for ls in self.lanes.lane_sections:
            lane = ls.center_lanes[0]
            lane._ref_line = substring(center_lane, ls.start_distance / ref_line.length,
                                       (ls.start_distance + ls.length) / ref_line.length,
                                       normalized=True)

    @property
    def boundary(self):
        """ Get the outer boundary of the road with all lanes """
        return self._boundary

    @property
    def elevation_profile(self) -> ElevationProfile:
        return self._elevation_profile

    @property
    def lateral_profile(self) -> LateralProfile:
        return self._lateral_profile

    @property
    def lanes(self) -> Lanes:
        """ Container object for all LaneSections of the road"""
        return self._lanes

    @property
    def drivable(self) -> bool:
        """ True if at least one lane is drivable in the road. """
        return any([ls.drivable for ls in self.lanes.lane_sections])

    @property
    def all_lane_forwards(self) -> bool:
        """ True if all lanes are forwards directed on the road (e.g., all-right lanes in right-handed traffic). """
        return all([all([ll.id < 0 for ll in ls.all_lanes if ll.id != 0]) for ls in self.lanes.lane_sections])

    @property
    def all_lane_backwards(self) -> bool:
        """ True if all lanes are backwards directed on the road (e.g., all-left lanes in right-handed traffic). """
        return all([all([ll.id > 0 for ll in ls.all_lanes if ll.id != 0]) for ls in self.lanes.lane_sections])
