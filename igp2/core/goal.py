import abc
import numpy as np
from typing import List, Optional
from shapely.geometry import Point, Polygon

from igp2.core.trajectory import Trajectory
from igp2.core.util import Box
from igp2.opendrive.elements.road_lanes import Lane


class Goal(abc.ABC):
    def __init__(self):
        self._center = None

    @abc.abstractmethod
    def reached(self, point: np.ndarray) -> bool:
        """ Returns whether a point is within the goal box / threshold radius. 
        Goal boundary is inclusive. """
        raise NotImplementedError

    @abc.abstractmethod
    def distance(self, point: np.ndarray) -> float:
        """ Calculate the distance of the given point to the Goal"""
        raise NotImplementedError

    @abc.abstractmethod
    def point_on_lane(self, lane: Lane) -> Optional[np.ndarray]:
        """ Return the closest point to the goal on the given lane midline."""
        raise NotImplementedError

    @abc.abstractmethod
    def passed_through_goal(self, trajectory: Trajectory) -> bool:
        """ Calculate whether the given trajectory has passed through a goal or not. """
        raise NotImplementedError

    @property
    def center(self) -> np.ndarray:
        """ Returns goal center point """
        return self._center


class PointGoal(Goal):
    """ A goal represented as a circle of with given threshold around a point."""
    def __init__(self, point: np.ndarray, threshold: float):
        super().__init__()
        self._radius = threshold
        self._center = point

    def __repr__(self):
        return f"PointGoal(center={np.round(self._center, 3)}, r={self._radius})"

    def reached(self, point: np.ndarray) -> bool:
        return self.distance(Point(point)) <= self._radius

    def distance(self, point: np.ndarray) -> float:
        return Point(point).distance(Point(self._center))

    def point_on_lane(self, lane: Lane) -> Optional[np.ndarray]:
        if lane.boundary.contains(Point(self._center)):
            distance = lane.distance_at(self._center)
            return lane.point_at(distance)
        return None

    def passed_through_goal(self, trajectory: Trajectory) -> bool:
        distances = np.linalg.norm(trajectory.path - self._center, axis=1)
        return np.any(np.isclose(distances, 0.0, atol=self.radius))

    @property
    def radius(self) -> float:
        """ Threshold radius"""
        return self._radius


class StoppingGoal(PointGoal):
    """ Subclass PointGoal to represent a stopping goal."""
    def __repr__(self):
        return f"StoppingGoal(center={np.round(self._center, 3)}, r={self._radius})"


class BoxGoal(Goal):
    """ A goal specified with a rectangle. """
    def __init__(self, box: Box):
        super().__init__()
        self._box = box
        self._poly = Polygon(box.boundary)
        self._center = box.center

    def __repr__(self):
        bounds_rep = str(np.round(np.array(self._poly.boundary.coords[0]), 3)).replace('\n', ' ')
        return f"BoxGoal(center={self._center}, bounds={bounds_rep})"

    def reached(self, point: np.ndarray) -> bool:
        point = Point(point)
        return self._poly.contains(point) or self._poly.touches(point)

    def distance(self, point: np.ndarray) -> float:
        return self._poly.distance(Point(point))

    def point_on_lane(self, lane: Lane) -> Optional[np.ndarray]:
        """ Return the point closest to the box center on the given lane midline."""
        if lane.midline.intersects(self._poly):
            distance = lane.distance_at(self.center)
            return lane.point_at(distance)
        return None

    def passed_through_goal(self, trajectory: Trajectory) -> bool:
        return any([self.reached(p) for p in trajectory.path])

    @property
    def box(self) -> Box:
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

    def distance(self, point: np.ndarray) -> float:
        return np.min([g.distance(point) for g in self._goals])

    def point_on_lane(self, lane: Lane) -> Optional[np.ndarray]:
        """ Returns a point on the given lane that is closest to one of the goal points in the collection. """
        closest_goal = None
        closest_distance = np.inf
        for goal in self._goals:
            distance = lane.midline.distance(Point(goal.center))
            if distance < closest_distance:
                closest_goal = goal
                closest_distance = distance
        return closest_goal.point_on_lane(lane)

    def passed_through_goal(self, trajectory: Trajectory) -> bool:
        for goal in self._goals:
            if goal.passed_through_goal(trajectory):
                return True
        return False

    @property
    def center(self) -> np.ndarray:
        """ The center of the point collection goal is defined as the point of the centers
        of the goals in the collection."""
        return np.mean([g.center for g in self._goals], axis=0)

    @property
    def radius(self):
        """ Maximum radius among each goal in the collection. """
        return max([g.radius for g in self._goals])

    def goals(self) -> List[PointGoal]:
        """ Return the list of PointGoals in this goal collection."""
        return self._goals
