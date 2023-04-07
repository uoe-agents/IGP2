from shapely.geometry.base import BaseGeometry

import igp2 as ip
import abc
import numpy as np
from typing import Union, List
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

    @abc.abstractmethod
    def passed_through_goal(self, trajectory: ip.Trajectory) -> bool:
        """ Calculate whether the given trajectory has passed through a goal or not. """
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
        return f"PointGoal(center={np.round(np.array(self._center.coords[0]), 3)}, r={self._radius})"

    def reached(self, point: np.ndarray) -> bool:
        return self.distance(Point(point)) <= self._radius

    def distance(self, point: BaseGeometry) -> float:
        return point.distance(self.center)

    def point_on_lane(self, lane: ip.Lane) -> Point:
        if lane.boundary.contains(self.center):
            distance = lane.distance_at(self.center)
            return Point(lane.point_at(distance))
        return Point()

    def passed_through_goal(self, trajectory: ip.Trajectory) -> bool:
        distances = np.linalg.norm(trajectory.path - self.center, axis=1)
        return np.any(np.isclose(distances, 0.0, atol=self.radius))

    @property
    def radius(self) -> float:
        """ Threshold radius"""
        return self._radius


class StoppingGoal(PointGoal):
    """ Subclass PointGoal to represent a stopping goal."""
    pass


class BoxGoal(Goal):
    """ A goal specified with a rectangle. """
    def __init__(self, box: ip.Box):
        super().__init__()
        self._box = box
        self._poly = Polygon(box.boundary)
        self._center = Point(box.center)

    def __repr__(self):
        bounds_rep = str(np.round(np.array(self._poly.boundary.coords[0]), 3)).replace('\n', ' ')
        return f"BoxGoal(center={self._center}, bounds={bounds_rep})"

    def reached(self, point: np.ndarray) -> bool:
        point = Point(point)
        return self._poly.contains(point) or self._poly.touches(point)

    def distance(self, point: BaseGeometry) -> float:
        return self.poly.distance(point)

    def point_on_lane(self, lane: ip.Lane) -> Point:
        """ Return the point closest to the box center on the given lane midline."""
        if lane.midline.intersects(self.poly):
            distance = lane.distance_at(self.center)
            return Point(lane.point_at(distance))
        return Point()

    def passed_through_goal(self, trajectory: ip.Trajectory) -> bool:
        return any([self.reached(p) for p in trajectory.path])

    @property
    def box(self) -> ip.Box:
        """ The box defining the goal"""
        return self._box

    @property
    def poly(self) -> Polygon:
        """ The Polygon describing the bounding box of the Goal."""
        return self._poly


class PointCollectionGoal(Goal):
    """ A goal that consists of a collection of PointGoals. """
    def __init__(self, goals: List[PointGoal]):
        super(PointCollectionGoal, self).__init__()
        self._goals = goals

    def __repr__(self):
        return f"PointCollectionGoal(n={len(self._goals)}, center={self._center})"

    def reached(self, point: np.ndarray) -> bool:
        for goal in self._goals:
            if goal.reached(point):
                return True
        return False

    def distance(self, point: BaseGeometry) -> float:
        return np.min([g.distance(point) for g in self._goals])

    def point_on_lane(self, lane: ip.Lane) -> Point:
        """ Returns a point on the given lane that is closest one of the goal points in the collection. """
        closest_goal = None
        closest_distance = np.inf
        for goal in self._goals:
            distance = goal.distance(lane.midline)
            if distance < closest_distance:
                closest_goal = goal
                closest_distance = distance
        return closest_goal.point_on_lane(lane)

    def passed_through_goal(self, trajectory: ip.Trajectory) -> bool:
        for goal in self._goals:
            if goal.passed_through_goal(trajectory):
                return True
        return False

    @property
    def center(self) -> Point:
        """ The center of the point collection goal is defined as the point of the centers
        of the goals in the collection."""
        return Point(np.mean([np.array(g.center.coords[0]) for g in self._goals], axis=0))

    def goals(self) -> List[PointGoal]:
        """ Return the list of PointGoals in this goal collection."""
        return self._goals
