from typing import Union

import numpy as np
from igp2.util import Box, get_curvature
from shapely.geometry import Point, Polygon
import abc


class Goal(abc.ABC):
    def __init__(self):
        self._center = None

    @abc.abstractmethod
    def reached(self, point: Point) -> bool:
        """ Returns whether a point is within the goal box / threshold radius. 
        Goal boundary is inclusive. """
        raise NotImplementedError

    @property
    def center(self) -> np.ndarray:
        """ Returns goal center point """
        return self._center


class PointGoal(Goal):
    def __init__(self, point: Union[np.ndarray, Point], threshold: float):
        super().__init__()
        self._point = point
        self._radius = threshold
        self._center = np.array([self._point.x, self._point.y]) if isinstance(point, Point) else point

    def __repr__(self):
        return f"PointGoal(center={self._center}, r={self._radius})"

    def reached(self, point: Point) -> bool:
        coord = np.array([point.x, point.y])
        diff = np.subtract(self._center, coord)
        return np.linalg.norm(diff) <= self._radius


class BoxGoal(Goal):
    def __init__(self, box: Box):
        super().__init__()
        self._poly = Polygon(box.boundary)
        self._center = box.center

    def __repr__(self):
        return f"BoxGoal(center={self._center}, bounds={list(self._poly.coords)})"

    def reached(self, point: Point) -> bool:
        return self._poly.contains(point) or self._poly.touches(point)
