from shapely.geometry.base import BaseGeometry

import igp2 as ip
import abc
import numpy as np
from typing import Union
from shapely.geometry import Point, Polygon


class Goal(abc.ABC):
    def __init__(self):
        self._center = None

    @abc.abstractmethod
    def reached(self, point: np.ndarray) -> bool:
        """ Returns whether a point is within the goal box / threshold radius. 
        Goal boundary is inclusive. """
        raise NotImplementedError

    @abc.abstractmethod
    def distance(self, point: BaseGeometry) -> float:
        """ Calculate the distance of the given point to the Goal"""
        raise NotImplementedError

    @abc.abstractmethod
    def point_on_lane(self, lane: ip.Lane) -> Point:
        """ Return the closest point to the goal on the given lane."""
        raise NotImplementedError

    @property
    def center(self) -> Point:
        """ Returns goal center point """
        return self._center


class PointGoal(Goal):
    """ A goal represented as a circle of with given threshold around a point."""
    def __init__(self, point: Union[np.ndarray, Point], threshold: float):
        super().__init__()
        self._point = point
        self._radius = threshold
        self._center = Point(point)

    def __repr__(self):
        return f"PointGoal(center={self._center}, r={self._radius})"

    def reached(self, point: np.ndarray) -> bool:
        return self.distance(Point(point)) <= self._radius

    def distance(self, point: BaseGeometry) -> float:
        return point.distance(self.center)

    def point_on_lane(self, lane: ip.Lane) -> Point:
        center = Point(self.center)
        if lane.boundary.contains(center):
            distance = lane.distance_at(center)
            return Point(lane.point_at(distance))
        return Point()

    @property
    def radius(self) -> float:
        """ Threshold radius"""
        return self._radius


class BoxGoal(Goal):
    """ A goal specified with a rectangle. """
    def __init__(self, box: ip.Box):
        super().__init__()
        self._box = box
        self._poly = Polygon(box.boundary)
        self._center = box.center

    def __repr__(self):
        return f"BoxGoal(center={self._center}, bounds={list(self._poly.boundary.coords)})"

    def reached(self, point: np.ndarray) -> bool:
        point = Point(point)
        return self._poly.contains(point) or self._poly.touches(point)

    def distance(self, point: BaseGeometry) -> float:
        return self.poly.distance(point)

    def point_on_lane(self, lane: ip.Lane) -> Point:
        """ Return the point closest to the box center on the given lane."""
        if lane.midline.intersects(self.poly):
            distance = lane.distance_at(self.center)
            return Point(lane.point_at(distance))
        return Point()

    @property
    def box(self) -> ip.Box:
        """ The box defining the goal"""
        return self._box

    @property
    def poly(self) -> Polygon:
        """ The Polygon describing the bounding box of the Goal."""
        return self._poly
