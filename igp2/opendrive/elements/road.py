# -*- coding: utf-8 -*-
import logging
import numpy as np

from shapely.geometry import JOIN_STYLE
from shapely.ops import unary_union
from shapely.geometry.polygon import Polygon

from igp2.opendrive.elements.geometry import cut_segment
from igp2.opendrive.elements.road_plan_view import PlanView
from igp2.opendrive.elements.road_link import Link
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
        self._link = Link()
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
    def midline(self):
        return self.plan_view.midline

    def calculate_boundary(self, fix_eps: float = 1e-2):
        """ Calculate the boundary Polygon of the road.
        Calculates boundaries of lanes as a sub-function.

        Args:
            fix_eps: If positive, then the algorithm attempts to fix sliver geometry in the map with this threshold
        """
        if self.lanes is None or self.lanes.lane_sections == []:
            return

        boundary = Polygon()
        for lane_section in self.lanes.lane_sections:
            start_line = cut_segment(self.midline,
                                     lane_section.start_distance,
                                     lane_section.start_distance + lane_section.length)
            prev_dir = None
            for lane in lane_section.all_lanes:
                current_dir = np.sign(lane.id)
                if prev_dir is None or prev_dir != current_dir:
                    ref_line = start_line
                lane_boundary, ref_line = lane.calculate_boundary(ref_line)
                boundary = unary_union([boundary, lane_boundary])
                prev_dir = current_dir

        if fix_eps > 0.0:
            boundary = boundary.buffer(fix_eps, 1, join_style=JOIN_STYLE.mitre) \
                .buffer(-fix_eps, 1, join_style=JOIN_STYLE.mitre)

        if not boundary.boundary.geom_type == "LineString":
            logger.warning(f"Boundary of road ID {self.id} is not a closed a loop!")

        self._boundary = boundary

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
        return self._lanes
