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
            scenario_map:
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
                     lane_path: List[Lane], scenario_map: Map) -> np.ndarray:
        velocity = self.get_curvature_velocity(path)
        vehicle_in_front_id, vehicle_in_front_dist = self.get_vehicle_in_front(agent_id, frame, lane_path, scenario_map)
        if vehicle_in_front_id is not None and vehicle_in_front_dist < 15:
            max_vel = frame[vehicle_in_front_id].speed  # TODO what if this is zero?
            assert max_vel > 1e-4
            velocity = np.minimum(velocity, max_vel)
        return velocity

    def get_vehicle_in_front(self, agent_id: int, frame: Dict[int, AgentState], lane_path: List[Lane],
                             scenario_map: Map) -> Tuple[float, float]:
        # TODO implement this
        return None, None


class FollowLane(Maneuver):

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        state = frame[agent_id]
        lane_sequence = self._get_lane_sequence(scenario_map)
        points = self._get_points(state, lane_sequence)
        path = self._get_path(state, points)
        velocity = self.get_velocity(path, agent_id, frame, lane_sequence, scenario_map)
        return VelocityTrajectory(path, velocity)

    def _get_lane_sequence(self, scenario_map: Map) -> List[Lane]:
        # TODO replace mock with actual function
        lane_seq = [scenario_map.roads[1].lanes.lane_sections[0].get_lane(2),
                    scenario_map.roads[7].lanes.lane_sections[0].get_lane(-1),
                    scenario_map.roads[2].lanes.lane_sections[0].get_lane(-2)]
        return lane_seq

    def _get_points(self, state: AgentState, lane_sequence: List[Lane]):
        final_point = lane_sequence[-1].midline.coords[-1]
        midline_points = [p for l in lane_sequence for p in l.midline.coords[:-1]] + [final_point]
        lane_ls = LineString(midline_points)
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
