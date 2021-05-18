import numpy as np
from util import Box, get_curvature
from shapely.geometry import Point, Polygon
import abc

class Goal(abc.ABC):

    @abc.abstractmethod
    def reached(self, point: Point) -> bool:
        """ Returns whether a point is within the goal box / threshold radius. 
        Goal boundary is inclusive. """
        pass

    @property
    def center(self) -> np.ndarray:
        """ Returns goal centerpoint """
        return self._center

class PointGoal(Goal):

    def __init__(self, point: Point, threshold: float):
        self._point = point
        self._radius = threshold
        self._center = np.array([self._point.x, self._point.y])

    def reached(self, point: Point) -> bool:
        coord = np.array([point.x, point.y])
        diff = np.subtract(self._center, coord)
        if np.linalg.norm(diff) <= self._radius : return True
        else: return False

class BoxGoal(Goal):

    def __init__(self, box: Box):
        self._poly = Polygon(box.boundary)
        self._center = box.center

    def reached(self, point: Point) -> bool:
        if self._poly.contains(point) or self._poly.touches(point): return True
        else: return False

# Point test
# a = Point(-1,-1)
# b = Point(-1,-0.5)
# pointgoal = PointGoal(a, 1)

# print(pointgoal.reached(b))

#Box test
center = np.array([1,1])
a = Box(center, 1, 1, 0)
print(a.boundary)
b = Point(0.4,0.4)

boxgoal = BoxGoal(a)
print(boxgoal.reached(b))
print(boxgoal.center)