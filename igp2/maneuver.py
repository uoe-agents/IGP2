from abc import ABC
from typing import Union, Tuple, List, Dict
from shapely.geometry import Point
import numpy as np

from igp2.agent import AgentState
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


class FollowLane(Maneuver):

    def get_trajectory(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map) -> VelocityTrajectory:
        state = frame[agent_id]
        lane = scenario_map.best_lane_at(state.position, state.heading, drivable_only=True)

        # TODO get sequence of lanes
        termination_point = Point(self.config.termination_point)
        assert lane.boundary.contains(termination_point)
        print(lane.id)


