# -*- coding: utf-8 -*-

from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString
from shapely.ops import unary_union
from shapely.geometry.polygon import Polygon

from igp2.opendrive.elements.roadPlanView import PlanView
from igp2.opendrive.elements.roadLink import Link
from igp2.opendrive.elements.roadLanes import Lanes
from igp2.opendrive.elements.roadElevationProfile import (
    ElevationProfile,
)
from igp2.opendrive.elements.roadLateralProfile import LateralProfile
from igp2.opendrive.elements.junction import Junction


class Road:
    """ """

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
        self._elevationProfile = ElevationProfile()
        self._lateralProfile = LateralProfile()
        self._lanes = Lanes()

    def __eq__(self, other):
        return other.__class__ is self.__class__ and self.__dict__ == other.__dict__

    def __repr__(self):
        return f"Road from {self.plan_view.start_position} with length {self.plan_view.length}"

    @property
    def id(self):
        """ """
        return self._id

    @id.setter
    def id(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._id = int(value)

    @property
    def name(self):
        """ """
        return self._name

    @name.setter
    def name(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._name = str(value)

    @property
    def junction(self):
        """ """
        return self._junction

    @junction.setter
    def junction(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, Junction) and value is not None:
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
        """ """
        return self._planView

    @property
    def midline(self):
        return self.plan_view.midline

    def calculate_boundary(self):
        """ Calculate the boundary Polygon of the road.
        Calculates boundaries of lanes as a sub-function.
        """
        if self.lanes is None or self.lanes.lane_sections == []:
            return

        boundary = Polygon()
        for lane_section in self.lanes.lane_sections:
            ref_line = self.midline
            for left_lane in lane_section.left_lanes:
                lane_boundary, ref_line = left_lane.calculate_boundary(ref_line)
                boundary = unary_union([boundary, lane_boundary])

            ref_line = self.midline
            for right_lane in lane_section.right_lanes:
                lane_boundary, ref_line = right_lane.calculate_boundary(ref_line)
                boundary = unary_union([boundary, lane_boundary])

        assert isinstance(boundary.boundary, LineString)
        self._boundary = boundary

    @property
    def boundary(self):
        """ Get the outer boundary of the road with all lanes """
        return self._boundary

    @property
    def elevation_profile(self) -> ElevationProfile:
        """ """
        return self._elevationProfile

    @property
    def lateral_profile(self) -> LateralProfile:
        """ """
        return self._lateralProfile

    @property
    def lanes(self) -> Lanes:
        """ """
        return self._lanes
