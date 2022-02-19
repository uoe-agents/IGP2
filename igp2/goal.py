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

    @property
    def center(self) -> np.ndarray:
        """ Returns goal center point """
        return self._center


class PointGoal(Goal):
    """ A goal represented as a circle of with given threshold around a point."""
    def __init__(self, point: Union[np.ndarray, Point], threshold: float):
        super().__init__()
        self._point = point
        self._radius = threshold
        self._center = np.array([self._point.x, self._point.y]) if isinstance(point, Point) else point

    def __repr__(self):
        return f"PointGoal(center={self._center}, r={self._radius})"

    def reached(self, point: np.ndarray) -> bool:
        diff = np.subtract(self._center, point)
        return np.linalg.norm(diff) <= self._radius

    @property
    def radius(self) -> float:
        """ Threshold radius"""
        return self._radius


class BoxGoal(Goal):
    """ A goal specified with a rectangle. """
    def __init__(self, box: ip.Box):
        super().__init__()
        self._poly = Polygon(box.boundary)
        self._center = box.center

    def __repr__(self):
        return f"BoxGoal(center={self._center}, bounds={list(self._poly.coords)})"

    def reached(self, point: np.ndarray) -> bool:
        point = Point(point)
        return self._poly.contains(point) or self._poly.touches(point)
