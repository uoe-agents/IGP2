from abc import ABC
from typing import Union, Tuple, List, Dict

from scipy.interpolate import CubicSpline
from shapely.geometry import Point, LineString
from shapely.ops import split
import numpy as np

from igp2.agent import AgentState
from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map
from igp2.trajectory import VelocityTrajectory
from igp2.util import get_curvature


class ManeuverConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    @property
    def type(self):
        return self.config_dict.get('type')

    @property
    def termination_point(self):
        return self.config_dict.get('termination_point', None)

    @property
    def exit_lane_id(self):
        return self.config_dict.get('exit_lane_id', None)


class Maneuver(ABC):

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

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        raise NotImplementedError

    @classmethod
    def get_curvature_velocity(cls, path: np.ndarray):
        c = np.abs(get_curvature(path))
        v = np.maximum(cls.MIN_SPEED, cls.MAX_SPEED * (1 - 3 * np.abs(c)))
        return v

    def get_velocity(self, path: np.ndarray, agent_id: int, frame: Dict[int, AgentState],
                     lane_path: List[Lane]) -> np.ndarray:
        velocity = self.get_curvature_velocity(path)
        vehicle_in_front_id, vehicle_in_front_dist = self.get_vehicle_in_front(agent_id, frame, lane_path)
        if vehicle_in_front_id is not None and vehicle_in_front_dist < 15:
            max_vel = frame[vehicle_in_front_id].speed  # TODO what if this is zero?
            assert max_vel > 1e-4
            velocity = np.minimum(velocity, max_vel)
        return velocity

    def get_vehicle_in_front(self, agent_id: int, frame: Dict[int, AgentState],
                             lane_path: List[Lane]) -> Tuple[float, float]:
        vehicles_in_path = self.get_vehicles_in_path(lane_path, frame)
        min_dist = np.inf
        vehicle_in_front = None
        state = frame[agent_id]

        # get linestring of lane midlines
        lane_ls = self.get_lane_path_midline(lane_path)
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
    def get_lane_path_midline(lane_path: List[Lane]) -> LineString:
        final_point = lane_path[-1].midline.coords[-1]
        midline_points = [p for l in lane_path for p in l.midline.coords[:-1]] + [final_point]
        lane_ls = LineString(midline_points)
        return lane_ls

    @staticmethod
    def get_vehicles_in_path(lane_path: List[Lane], frame: Dict[int, AgentState]) -> List[int]:
        agents = []
        for agent_id, agent_state in frame.items():
            point = Point(agent_state.position)
            for lane in lane_path:
                if lane.boundary.contains(point):
                    agents.append(agent_id)
        return agents


class FollowLane(Maneuver):

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        state = frame[agent_id]
        lane_sequence = self._get_lane_sequence(scenario_map)
        points = self._get_points(state, lane_sequence)
        path = self._get_path(state, points)
        velocity = self.get_velocity(path, agent_id, frame, lane_sequence)
        return VelocityTrajectory(path, velocity)

    def _get_lane_sequence(self, scenario_map: Map) -> List[Lane]:
        # TODO replace mock with actual function
        lane_seq = [scenario_map.roads[1].lanes.lane_sections[0].get_lane(2),
                    scenario_map.roads[7].lanes.lane_sections[0].get_lane(-1),
                    scenario_map.roads[2].lanes.lane_sections[0].get_lane(-2)]
        return lane_seq

    def _get_points(self, state: AgentState, lane_sequence: List[Lane]):
        lane_ls = self.get_lane_path_midline(lane_sequence)
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
            if point_lon > current_lon + margin and first_ls_point is None:
                first_ls_point = point
            if first_ls_point is not None:
                if point_lon + self.POINT_SPACING > termination_lon:
                    break
                else:
                    final_ls_point = point

        if first_ls_point is None or final_ls_point is None:
            raise Exception('Could not find first/final point')

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


class SwitchLane(Maneuver):
    TARGET_SITCH_LENGTH = 20
    MIN_SWITCH_LENGTH = 5

    def _get_lane_sequence(self, scenario_map: Map) -> List[Lane]:
        #TODO replace mock with actual
        return [scenario_map.roads[1].lanes.lane_sections[0].get_lane(1)]

    def _get_path(self, state: AgentState, scenario_map: Map) -> np.ndarray:
        final_lane = scenario_map.best_lane_at(self.config.termination_point, drivable_only=True)
        initial_point = state.position
        target_point = np.array(self.config.termination_point)
        final_lon = final_lane.midline.project(Point(initial_point))
        dist = np.linalg.norm(target_point - initial_point)
        initial_direction = np.array([np.cos(state.heading), np.sin(state.heading)])
        target_direction = final_lane.get_direction_at(final_lon)

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
        lane_sequence = self._get_lane_sequence(scenario_map)
        path = self._get_path(state, scenario_map)
        velocity = self.get_velocity(path, agent_id, frame, lane_sequence)
        return VelocityTrajectory(path, velocity)
