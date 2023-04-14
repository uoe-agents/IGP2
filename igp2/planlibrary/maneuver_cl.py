import igp2 as ip
import abc
import numpy as np
import logging
from typing import Dict
from shapely.geometry import LineString, Point
from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig, FollowLane, Turn, \
    GiveWay, SwitchLaneLeft, SwitchLaneRight, TrajectoryManeuver
from igp2.planlibrary.controller import PIDController, AdaptiveCruiseControl
from igp2.agentstate import AgentState

logger = logging.getLogger(__name__)


class ClosedLoopManeuver(Maneuver, abc.ABC):
    """ Defines a maneuver in which sensor feedback is used """

    def next_action(self, observation: ip.Observation) -> ip.Action:
        """ Selects the next action for the vehicle to take

        Args:
            observation: current environment Observation

        Returns:
            Action that the vehicle should take
        """
        raise NotImplementedError

    def done(self, observation: ip.Observation) -> bool:
        """ Checks if the maneuver is finished

        Args:
            observation: current environment Observation


        Returns:
            Bool indicating whether the maneuver is completed
        """
        raise NotImplementedError


class WaypointManeuver(ClosedLoopManeuver, abc.ABC):
    WAYPOINT_MARGIN = 1
    COMPLETION_MARGIN = 0.5
    LATERAL_ARGS = {'K_P': 1.95, 'K_I': 0.2, 'K_D': 0.0}
    LONGITUDINAL_ARGS = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0}
    ACC_ARGS = {'a_a': 5, 'b_a': 5, 'delta': 4., 's_0': 2., 'T_a': 1.5}
    FPS = 20

    def __init__(self,
                 config: ManeuverConfig,
                 agent_id: int,
                 frame: Dict[int, ip.AgentState],
                 scenario_map: ip.Map):
        super().__init__(config, agent_id, frame, scenario_map)
        self._controller = PIDController(1 / self.FPS, self.LATERAL_ARGS, self.LONGITUDINAL_ARGS)
        self._acc = AdaptiveCruiseControl(1 / self.FPS, **self.ACC_ARGS)

    def get_target_waypoint(self, state: ip.AgentState):
        """ Get the index of the target waypoint in the reference trajectory"""
        dist = np.linalg.norm(self.trajectory.path - state.position, axis=1)
        closest_idx = np.argmin(dist)
        if dist[-1] < self.WAYPOINT_MARGIN:
            target_wp_idx = len(self.trajectory.path) - 1
        else:
            far_waypoints_dist = dist[closest_idx:]
            target_wp_idx = closest_idx + np.argmax(far_waypoints_dist >= self.WAYPOINT_MARGIN)
        return target_wp_idx, closest_idx

    def next_action(self, observation: ip.Observation) -> ip.Action:
        target_wp_idx, closest_idx = self.get_target_waypoint(observation.frame[self.agent_id])
        target_waypoint = self.trajectory.path[target_wp_idx]
        target_velocity = self.trajectory.velocity[closest_idx]
        return self._get_action(target_waypoint, target_velocity, observation)

    def _get_action(self, target_waypoint: np.ndarray, target_velocity: float, observation: ip.Observation):
        velocity_error = self._get_acceleration(target_velocity, observation.frame)
        heading_error = self._get_steering(target_waypoint, observation.frame)

        acceleration, steering = self._controller.next_action(velocity_error, heading_error)

        action = ip.Action(acceleration, steering, target_velocity)
        return action

    def _get_steering(self, target_waypoint: np.ndarray, frame: Dict[int, AgentState]) -> float:
        state = frame[self.agent_id]
        target_direction = target_waypoint - state.position
        waypoint_heading = np.arctan2(target_direction[1], target_direction[0])
        if np.all(target_waypoint == self.trajectory.path[-1]):
            waypoint_heading = self.trajectory.heading[-1]
        heading_error = np.diff(np.unwrap([state.heading, waypoint_heading]))[0]
        return heading_error

    def _get_acceleration(self, target_velocity: float, frame: Dict[int, ip.AgentState]):
        state = frame[self.agent_id]
        acceleration = target_velocity - state.speed
        vehicle_in_front, dist = self.get_vehicle_in_front(frame, self.lane_sequence)
        if vehicle_in_front is not None:
            in_front_speed = frame[vehicle_in_front].speed
            gap = dist - state.metadata.length
            acc_acceleration = self._acc.get_acceleration(self.MAX_SPEED, state.speed, in_front_speed, gap)
            acceleration = min(acceleration, acc_acceleration)

        return acceleration

    def done(self, observation: ip.Observation) -> bool:
        state = observation.frame[self.agent_id]
        ls = LineString(self.trajectory.path)
        p = Point(state.position)
        dist_along = ls.project(p)
        dist_from_end = np.linalg.norm(state.position - self.trajectory.path[-1])
        # We want the vehicle to enter the next lane so we are not done until we have not passed the midline
        ret = dist_along >= ls.length and dist_from_end > self.COMPLETION_MARGIN
        return ret


class FollowLaneCL(FollowLane, WaypointManeuver):
    """ Closed loop follow lane maneuver """
    pass


class TurnCL(Turn, WaypointManeuver):
    """ Closed loop turn maneuver """
    pass


class SwitchLaneLeftCL(SwitchLaneLeft, WaypointManeuver):
    """ Closed loop switch lane left maneuver """
    pass


class SwitchLaneRightCL(SwitchLaneRight, WaypointManeuver):
    """ Closed loop switch lane right maneuver """
    pass


class TrajectoryManeuverCL(TrajectoryManeuver, WaypointManeuver):
    """ Closed loop maneuver that follows a pre-defined trajectory """
    pass


class GiveWayCL(GiveWay, WaypointManeuver):
    """ Closed loop give way maneuver """

    def __stop_required(self, observation: ip.Observation, target_wp_idx: int):
        ego_time_to_junction = self.trajectory.times[-1] - self.trajectory.times[target_wp_idx]
        times_to_junction = self._get_times_to_junction(
            observation.frame, observation.scenario_map, ego_time_to_junction)
        time_until_clear = self._get_time_until_clear(ego_time_to_junction, times_to_junction)
        return time_until_clear > 0

    def next_action(self, observation: ip.Observation) -> ip.Action:
        state = observation.frame[self.agent_id]
        target_wp_idx, closest_idx = self.get_target_waypoint(state)
        target_waypoint = self.trajectory.path[target_wp_idx]
        dist_to_junction = np.linalg.norm(self.trajectory.path[-1] - state.position)
        # Based on d = v^2 / (2 * mu * g), with mu=0.45 which is corresponds to a wet road friction coefficient
        stopping_distance = state.speed ** 2 / (2 * 0.5 * 9.8) + state.metadata.length / 2
        close_to_junction_entry = dist_to_junction < stopping_distance

        target_velocity = max(self.trajectory.velocity[target_wp_idx], self.STANDBY_VEL)
        if close_to_junction_entry and \
                self.config.stop and \
                self.__stop_required(observation, target_wp_idx):
            target_velocity = 0
        return self._get_action(target_waypoint, target_velocity, observation)


class CLManeuverFactory:
    maneuver_types = {"follow-lane": FollowLaneCL,
                      "switch-left": SwitchLaneLeftCL,
                      "switch-right": SwitchLaneRightCL,
                      "turn": TurnCL,
                      "give-way": GiveWayCL,
                      "trajectory": TrajectoryManeuverCL}

    @classmethod
    def create(cls, config: ManeuverConfig, agent_id: int, frame: Dict[int, ip.AgentState], scenario_map: ip.Map):
        config.config_dict["adjust_swerving"] = False
        return cls.maneuver_types[config.type](config, agent_id, frame, scenario_map)
