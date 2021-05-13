import abc
import logging
from abc import ABC
from typing import Union, Tuple, List, Dict

from scipy.interpolate import CubicSpline
from shapely.geometry import Point, LineString
from shapely.ops import split
import numpy as np

from igp2.agent import AgentState
from igp2.opendrive.elements.road_lanes import Lane, LaneTypes
from igp2.opendrive.map import Map
from igp2.trajectory import VelocityTrajectory
from igp2.util import get_curvature

logger = logging.getLogger(__name__)


class ManeuverConfig:
    """ Contains the parameters describing a maneuver """

    def __init__(self, config_dict):
        """ Define a ManeuverConfig object which describes the configuration of a maneuver

        Args:
            config_dict: dictionary containing parameters of the maneuver
        """
        self.config_dict = config_dict

    @property
    def type(self) -> str:
        """ The type of the maneuver.
        Acceptable values are {'follow-lane', 'switch-left', 'switch-right', 'turn', 'give-way'}
        """
        return self.config_dict.get('type')

    @property
    def termination_point(self) -> Tuple[float, float]:
        """ Point at which the maneuver trajectory terminates """
        return self.config_dict.get('termination_point', None)

    @property
    def junction_road_id(self) -> int:
        """ Used for give-way and turn """
        return self.config_dict.get('junction_road_id', None)

    @property
    def junction_lane_id(self) -> int:
        """ Used for give-way and turn"""
        return self.config_dict.get('junction_lane_id', None)


class Maneuver(ABC):
    """ Abstract class for a vehicle maneuver """
    POINT_SPACING = 1
    MAX_SPEED = 10
    MIN_SPEED = 3

    def __init__(self, config: ManeuverConfig, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map):
        """ Create a maneuver object along with it's target trajectory

        Args:
            config: Parameters of the maneuver
            agent_id: identifier for the agent
            frame: dictionary containing state of all observable agents
            scenario_map: local road map
        """
        self.config = config
        self.trajectory = self.get_trajectory(agent_id, frame, scenario_map)

    @abc.abstractmethod
    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        """ Generates the target trajectory for the maneuver

        Args:
            agent_id: identifier for the agent
            frame: dictionary containing state of all observable agents
            scenario_map: local road map

        Returns:
            Target trajectory
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Checks whether the maneuver is applicable for an agent

        Args:
            state: current state of the agent
            scenario_map: local road map

        Returns:
            Boolean value indicating whether the maneuver is applicable
        """
        raise NotImplementedError

    @classmethod
    def get_curvature_velocity(cls, path: np.ndarray) -> np.ndarray:
        """ Generate target velocities based on the curvature of the road """
        c = np.abs(get_curvature(path))
        v = np.maximum(cls.MIN_SPEED, cls.MAX_SPEED * (1 - 3 * np.abs(c)))
        return v

    def get_velocity(self, path: np.ndarray, agent_id: int, frame: Dict[int, AgentState],
                     lane_path: List[Lane]) -> np.ndarray:
        """ Generate target velocities based on the curvature of the path and vehicle in front.

        Args:
            path: target path along which the agent will travel
            agent_id: identifier for the agent
            frame: dictionary containing state of all observable agents
            lane_path: sequence of lanes that the agent will travel along

        Returns:
            array of target velocities
        """
        velocity = self.get_curvature_velocity(path)
        vehicle_in_front_id, vehicle_in_front_dist = self.get_vehicle_in_front(agent_id, frame, lane_path)
        if vehicle_in_front_id is not None and vehicle_in_front_dist < 15:
            max_vel = frame[vehicle_in_front_id].speed  # TODO what if this is zero?
            assert max_vel > 1e-4
            velocity = np.minimum(velocity, max_vel)
        return velocity

    @classmethod
    def get_vehicle_in_front(cls, agent_id: int, frame: Dict[int, AgentState],
                             lane_path: List[Lane]) -> Tuple[int, float]:
        """ Finds the vehicle in front of an agent.

        Args:
            agent_id: identifier for the agent
            frame: dictionary containing state of all observable agents
            lane_path: sequence of lanes that the agent will travel along

        Returns:
            vehicle_in_front: ID for the agent in front
            dist: distance to the vehicle in front
        """
        vehicles_in_path = cls._get_vehicles_in_path(lane_path, frame)
        min_dist = np.inf
        vehicle_in_front = None
        state = frame[agent_id]

        # get linestring of lane midlines
        lane_ls = cls._get_lane_path_midline(lane_path)
        ego_lon = lane_ls.project(Point(state.position))

        # find vehicle in front with closest distance
        for agent_id in vehicles_in_path:
            agent_lon = lane_ls.project(Point(frame[agent_id].position))
            dist = agent_lon - ego_lon
            if 0 < dist < min_dist:
                vehicle_in_front = agent_id
                min_dist = dist
        return vehicle_in_front, min_dist

    @staticmethod
    def _get_lane_path_midline(lane_path: List[Lane]) -> LineString:
        final_point = lane_path[-1].midline.coords[-1]
        midline_points = [p for l in lane_path for p in l.midline.coords[:-1]] + [final_point]
        lane_ls = LineString(midline_points)
        return lane_ls

    @staticmethod
    def _get_vehicles_in_path(lane_path: List[Lane], frame: Dict[int, AgentState]) -> List[int]:
        agents = []
        for agent_id, agent_state in frame.items():
            point = Point(agent_state.position)
            for lane in lane_path:
                if lane.boundary.contains(point):
                    agents.append(agent_id)
        return agents


class FollowLane(Maneuver):
    """ Defines a follow-lane maneuver """

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        state = frame[agent_id]
        lane_sequence = self._get_lane_sequence(state, scenario_map)
        points = self._get_points(state, lane_sequence)
        path = self._get_path(state, points)
        velocity = self.get_velocity(path, agent_id, frame, lane_sequence)
        return VelocityTrajectory(path, velocity)

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Checks whether the follow lane maneuver is applicable for an agent.
            Follow lane is applicable if the agent is in a drivable lane.

        Args:
            state: Current state of the agent
            scenario_map: local road map

        Returns:
            Boolean indicating whether the maneuver is applicable
        """
        return len(scenario_map.lanes_at(state.position, drivable_only=True)) > 0

    def _get_lane_sequence(self, state: AgentState, scenario_map: Map) -> List[Lane]:
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        lane_seq = [current_lane]
        return lane_seq

    def _get_points(self, state: AgentState, lane_sequence: List[Lane]):
        lane_ls = self._get_lane_path_midline(lane_sequence)
        current_point = Point(state.position)
        termination_lon = lane_ls.project(Point(self.config.termination_point))
        termination_point = lane_ls.interpolate(termination_lon).coords[0]
        lat_dist = lane_ls.distance(current_point)
        current_lon = lane_ls.project(current_point)

        margin = self.POINT_SPACING + 2 * lat_dist

        assert current_lon < lane_ls.length - margin, 'current point is past end of lane'
        assert current_lon < termination_lon, 'current point is past the termination point'

        # trim out points we have passed
        first_ls_point = None
        final_ls_point = None

        for coord in lane_ls.coords:
            point = Point(coord)
            point_lon = lane_ls.project(point)
            if termination_lon - margin > point_lon > current_lon + margin and first_ls_point is None:
                first_ls_point = point
            if first_ls_point is not None:
                if point_lon + self.POINT_SPACING > termination_lon:
                    break
                else:
                    final_ls_point = point

        if first_ls_point is None:
            # none of the points are between start and termination position
            return np.array([state.position, termination_point])

        if final_ls_point == first_ls_point:
            trimmed_points = first_ls_point
        else:
            # trim out points before first point
            if first_ls_point == Point(lane_ls.coords[-1]):
                # handle case where first point is final point
                following_points = first_ls_point
            else:
                following_points = split(lane_ls, first_ls_point)[-1]
            # trim out points after final point
            trimmed_points = split(following_points, final_ls_point)[0]

        all_points = list(current_point.coords) + list(trimmed_points.coords) + [termination_point]
        return np.array(all_points)

    def _get_path(self, state: AgentState, points: np.ndarray):
        if len(points) == 2:
            return points

        heading = state.heading
        initial_direction = np.array([np.cos(heading), np.sin(heading)])
        final_direction = np.diff(points[-2:], axis=0).flatten()
        final_direction = final_direction / np.linalg.norm(final_direction)
        t = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        cs = CubicSpline(t, points, bc_type=((1, initial_direction), (1, final_direction)))
        num_points = int(t[-1] / self.POINT_SPACING)
        ts = np.linspace(0, t[-1], num_points)
        path = cs(ts)
        return path


class Turn(FollowLane):
    """ Defines a turn maneuver """

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Checks whether the turn maneuver is applicable for an agent.
            Turn is applicable if the agents current lane or next lane is in a junction.

        Args:
            state: Current state of the agent
            scenario_map: local road map

        Returns:
            Boolean indicating whether the maneuver is applicable
        """
        currently_in_junction = scenario_map.junction_at(state.position) is not None
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        next_lanes = current_lane.link.successor
        next_lane_is_junction = next_lanes is not None and any([l.parent_road.junction is not None for l in next_lanes])
        return currently_in_junction or next_lane_is_junction

    def _get_lane_sequence(self, state: AgentState, scenario_map: Map) -> List[Lane]:
        junction_lane = scenario_map.get_lane(self.config.junction_road_id, self.config.junction_lane_id)
        return [junction_lane]


class SwitchLane(Maneuver, ABC):
    """ Defines a switch lane maneuver """
    TARGET_SWITCH_LENGTH = 20
    MIN_SWITCH_LENGTH = 5

    def _get_path(self, state: AgentState, target_lane: Lane) -> np.ndarray:
        initial_point = state.position
        target_point = np.array(self.config.termination_point)
        final_lon = target_lane.midline.project(Point(initial_point))
        dist = np.linalg.norm(target_point - initial_point)
        initial_direction = np.array([np.cos(state.heading), np.sin(state.heading)])
        target_direction = target_lane.get_direction_at(final_lon)

        """
        Fit cubic curve given boundary conditions at t=0 and t=1
        boundary == A @ coeff
        coeff = inv(A) @ boundary
        A = array([[0, 0, 0, 1],
                   [1, 1, 1, 1],
                   [0, 0, 1, 0],
                   [3, 2, 1, 0]])
        transform = np.linalg.inv(A)
        """

        transform = np.array([[ 2., -2.,  1.,  1.],
                              [-3.,  3., -2., -1.],
                              [ 0.,  0.,  1.,  0.],
                              [ 1.,  0.,  0.,  0.]])

        boundary = np.vstack([initial_point,
                              target_point,
                              initial_direction * dist,
                              target_direction * dist])
        coeff = transform @ boundary

        # evaluate points on cubic curve
        num_points = max(2, int(dist / self.POINT_SPACING) + 1)
        t = np.linspace(0, 1, num_points)
        powers = np.power(t.reshape((-1, 1)), np.arange(3, -1, -1))
        points = powers @ coeff
        return points

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        state = frame[agent_id]
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        target_lane = scenario_map.best_lane_at(self.config.termination_point, drivable_only=True)

        assert current_lane.parent_road == target_lane.parent_road, 'initial lane and target lane not in same road'
        assert abs(current_lane.id - target_lane.id) == 1, 'initial lane and target lane not adjacent'

        path = self._get_path(state, target_lane)
        velocity = self.get_velocity(path, agent_id, frame, [target_lane])
        return VelocityTrajectory(path, velocity)


class SwitchLaneLeft(SwitchLane):
    """ Defines a switch lane left maneuver"""

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Checks whether the switch lane left maneuver is applicable for an agent.
            Switch lane left is applicable if it is legal to switch to a lane left of the current lane.

        Args:
            state: Current state of the agent
            scenario_map: local road map

        Returns:
            Boolean indicating whether the maneuver is applicable
        """
        # TODO: Add check for lane marker
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        left_lane = current_lane.lane_section.get_lane(current_lane.id - 1)
        return (left_lane is not None
                and left_lane.type == LaneTypes.DRIVING
                and (current_lane.id < 0) == (left_lane.id < 0))


class SwitchLaneRight(SwitchLane):

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Checks whether the switch right left maneuver is applicable for an agent.
            Switch lane right is applicable if it is legal to switch to a lane right of the current lane.

        Args:
            state: Current state of the agent
            scenario_map: local road map

        Returns:
            Boolean indicating whether the maneuver is applicable
        """
        # TODO: Add check for lane marker
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        right_lane = current_lane.lane_section.get_lane(current_lane.id + 1)
        return (right_lane is not None
                and right_lane.type == LaneTypes.DRIVING
                and (current_lane.id < 0) == (right_lane.id < 0))


class GiveWay(FollowLane):
    MAX_ONCOMING_VEHICLE_DIST = 100
    GAP_TIME = 3

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        state = frame[agent_id]
        lane_sequence = self._get_lane_sequence(state, scenario_map)
        points = self._get_points(state, lane_sequence)
        path = self._get_path(state, points)

        velocity = self._get_const_deceleration_vel(state.speed, 2, path)
        ego_time_to_junction = VelocityTrajectory(path, velocity).duration
        times_to_junction = self._get_times_to_junction(agent_id, frame, scenario_map, ego_time_to_junction)
        time_until_clear = self._get_time_until_clear(ego_time_to_junction, times_to_junction)
        stop_time = time_until_clear - ego_time_to_junction

        if stop_time > 0:
            # insert waiting points
            path = self._add_stop_points(path)
            velocity = self._add_stop_velocities(path, velocity, stop_time)
            
        return VelocityTrajectory(path, velocity)

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Checks whether the give way maneuver is applicable for an agent.
            Give way is applicable if the next lane is in a junction.

        Args:
            state: Current state of the agent
            scenario_map: local road map

        Returns:
            Boolean indicating whether the maneuver is applicable
        """
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        next_lanes = current_lane.link.successor
        return next_lanes is not None and any([l.parent_road.junction is not None for l in next_lanes])

    def _get_times_to_junction(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map,
                               ego_time_to_junction: float) -> List[float]:
        # get oncoming vehicles
        oncoming_vehicles = self._get_oncoming_vehicles(agent_id, frame, scenario_map)

        time_to_junction = []
        for agent, dist in oncoming_vehicles:
            # check if the vehicle is stopped
            time = dist / agent.speed
            if agent.speed > 1 and time > ego_time_to_junction - self.GAP_TIME:
                time_to_junction.append(time)

        return time_to_junction

    def _get_oncoming_vehicles(self, ego_agent_id, frame: Dict[int, AgentState],
                               scenario_map: Map) -> List[Tuple[AgentState, float]]:
        oncoming_vehicles = []

        ego_junction_lane = scenario_map.get_lane(self.config.junction_road_id, self.config.junction_lane_id)
        lanes_to_cross = self._get_lanes_to_cross(scenario_map)

        agent_lanes = [(i, scenario_map.best_lane_at(s.position, s.heading, True)) for i, s in frame.items()]

        for lane_to_cross in lanes_to_cross:
            lane_sequence = self._get_predecessor_lane_sequence(lane_to_cross)
            midline = self._get_lane_path_midline(lane_sequence)
            crossing_point = lane_to_cross.boundary.intersection(ego_junction_lane.boundary).centroid
            crossing_lon = midline.project(crossing_point)

            # find agents in lane to cross
            for agent_id, agent_lane in agent_lanes:
                agent_state = frame[agent_id]
                if agent_id != ego_agent_id and agent_lane in lane_sequence:
                    agent_lon = midline.project(Point(agent_state.position))
                    dist = crossing_lon - agent_lon
                    if 0 < dist < self.MAX_ONCOMING_VEHICLE_DIST:
                        oncoming_vehicles.append((agent_state, dist))
        return oncoming_vehicles

    def _get_lanes_to_cross(self, scenario_map: Map) -> List[Lane]:
        ego_road = scenario_map.roads.get(self.config.junction_road_id)
        ego_lane = scenario_map.get_lane(self.config.junction_road_id, self.config.junction_lane_id)
        ego_incoming_lane = ego_lane.link.predecessor
        lanes = []
        for connection in ego_road.junction.connections:
            for lane_link in connection.lane_links:
                lane = lane_link.to_lane
                same_predecessor = (ego_incoming_lane.id == lane_link.from_id
                                    and ego_incoming_lane.parent_road.id == connection.incoming_road.id)
                if not (same_predecessor or self._has_priority(ego_road, lane.parent_road)):
                    overlap = ego_lane.boundary.intersection(lane.boundary)
                    if overlap.area > 1:
                        lanes.append(lane)
        return lanes

    @classmethod
    def _get_predecessor_lane_sequence(cls, lane: Lane) -> List[Lane]:
        lane_sequence = []
        total_length = 0
        while lane is not None and total_length < cls.MAX_ONCOMING_VEHICLE_DIST:
            lane_sequence.insert(0, lane)
            total_length += lane.midline.length
            lane = lane.link.predecessor
        return lane_sequence

    @staticmethod
    def _has_priority(ego_road, other_road):
        for priority in ego_road.junction.priorities:
            if (priority.high_id == ego_road.id
                    and priority.low_id == other_road.id):
                return True
        return False

    @classmethod
    def _get_time_until_clear(cls, ego_time_to_junction: float, times_to_junction: List[float]) -> float:
        if len(times_to_junction) == 0:
            return 0.
        times_to_junction = np.array(times_to_junction)
        times_to_junction = times_to_junction[times_to_junction >= ego_time_to_junction]
        times_to_junction = np.concatenate([[ego_time_to_junction], times_to_junction, [np.inf]])
        gaps = np.diff(times_to_junction)
        first_long_gap = np.argmax(gaps >= cls.GAP_TIME)
        return times_to_junction[first_long_gap]

    @staticmethod
    def _get_const_deceleration_vel(initial_vel, final_vel, path):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        velocity = initial_vel + s / s[-1] * (final_vel - initial_vel)
        return velocity

    @staticmethod
    def _add_stop_points(path):
        p_start = path[-2, None]
        p_end = path[-1, None]
        diff = p_end - p_start
        p_stop_frac = np.array([[0.7, 0.9]]).T
        p_stop = p_start + p_stop_frac @ diff
        new_path = np.concatenate([path[:-1], p_stop, p_end])
        return new_path

    @classmethod
    def _add_stop_velocities(cls, path, velocity, stop_time):
        stop_vel = cls._get_stop_velocity(path, velocity, stop_time)
        velocity = np.insert(velocity, -1, [stop_vel] * 2)
        return velocity

    @staticmethod
    def _get_stop_velocity(path, velocity, stop_time):
        # calculate stop velocities assuming constant acceleration in each segment
        final_section = path[-4:]
        s = np.linalg.norm(np.diff(final_section, axis=0), axis=1)
        v1, v2 = velocity[-2:]
        t = stop_time + 2 * np.sum(s) / (v1 + v2)
        A = np.array([[t, 0, 0, 0],
                      [t * (v1 + v2), -2, -1, -2],
                      [v1 * v2 * t, -2 * v2, -v1 - v2, -2 * v1],
                      [0, 0, -v1 * v2, 0]])
        coeff = A @ np.concatenate([[1], s]).T
        r = np.roots(coeff)
        stop_vel = np.max(r.real[np.abs(r.imag < 1e-5)])
        return stop_vel
