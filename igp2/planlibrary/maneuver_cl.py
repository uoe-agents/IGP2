import abc
from typing import Dict

import numpy as np
from shapely.geometry import LineString, Point

from igp2.agent.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig, FollowLane, Turn, SwitchLaneLeft, SwitchLaneRight, \
    TrajectoryManeuver, GiveWay
from igp2.vehicle import Observation, Action


class PController:
    """ Proportional controller """

    def __init__(self, kp=1):
        """ Defines a proportional controller object

        Args:
            kp: constant for proportional control term
        """
        self.kp = kp

    def next_action(self, error):
        return self.kp * error


class AdaptiveCruiseControl:
    """ Defines an adaptive cruise controller based on the intelligent driver model (IDM)"""

    def __init__(self, a_a=5, b_a=5, delta=4., s_0=2., T_a=1.5):
        """ Initialise the parameters of the adaptive cruise controller

        Args:
            a_a: maximum positive acceleration
            b_a: maximum negative acceleration
            delta: acceleration exponent
            s_0: minimum desired gap
            T_a: following time-gap
        """
        self.delta = delta
        self.s_0 = s_0
        self.a_a = a_a
        self.b_a = b_a
        self.T_a = T_a

    def get_acceleration(self, v_0: float, v_a: float, v_f: float, s_a: float) -> float:
        """ Get the acceleration output by the controller

        Args:
            v_0: maximum velocity
            v_a: ego vehicle velocity
            v_f: front vehicle velocity
            s_a: gap between vehicles

        Returns:
            acceleration
        """
        delta_v = v_a - v_f
        s_star = self.s_0 + self.T_a * v_a + v_a * delta_v / (2 * np.sqrt(self.a_a * self.b_a))
        accel = self.a_a * (1 - (v_a / v_0) ** self.delta - (s_star / s_a) ** 2)
        return accel


class ClosedLoopManeuver(Maneuver, abc.ABC):
    """ Defines a maneuver in which sensor feedback is used """

    def next_action(self, observation: Observation) -> Action:
        """ Selects the next action for the vehicle to take

        Args:
            observation: current environment Observation

        Returns:
            Action that the vehicle should take
        """
        raise NotImplementedError

    def done(self, observation: Observation) -> bool:
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

    def __init__(self, config: ManeuverConfig, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map):
        super().__init__(config, agent_id, frame, scenario_map)
        self._acceleration_controller = PController(1)
        self._steer_controller = PController(1)
        self._acc = AdaptiveCruiseControl()

    def get_target_waypoint(self, state: AgentState):
        """ Get the index of the target waypoint in the reference trajectory"""
        dist = np.linalg.norm(self.trajectory.path - state.position, axis=1)
        closest_idx = np.argmin(dist)
        if dist[-1] < self.WAYPOINT_MARGIN:
            target_wp_idx = len(self.trajectory.path) - 1
        else:
            far_waypoints_dist = dist[closest_idx:]
            target_wp_idx = closest_idx + np.argmax(far_waypoints_dist >= self.WAYPOINT_MARGIN)
        return target_wp_idx, closest_idx

    def next_action(self, observation: Observation) -> Action:
        # get target waypoint
        target_wp_idx, closest_idx = self.get_target_waypoint(observation.frame[self.agent_id])
        target_waypoint = self.trajectory.path[target_wp_idx]
        target_velocity = self.trajectory.velocity[closest_idx]
        return self._get_action(target_waypoint, target_velocity, observation)

    def _get_action(self, target_waypoint, target_velocity, observation):
        state = observation.frame[self.agent_id]
        target_direction = target_waypoint - state.position
        waypoint_heading = np.arctan2(target_direction[1], target_direction[0])
        heading_error = np.diff(np.unwrap([state.heading, waypoint_heading]))[0]
        if np.all(target_waypoint == self.trajectory.path[-1]):
            steer_angle = 0
        else:
            steer_angle = self._steer_controller.next_action(heading_error)
        acceleration = self._get_acceleration(target_velocity, observation.frame)
        action = Action(acceleration, steer_angle)
        return action

    def _get_acceleration(self, target_velocity: float, frame: Dict[int, AgentState]):
        state = frame[self.agent_id]
        pid_acceleration = self._acceleration_controller.next_action(target_velocity - state.speed)
        vehicle_in_front, dist = self.get_vehicle_in_front(frame, self.lane_sequence)
        if vehicle_in_front is None:
            acceleration = pid_acceleration
        else:
            in_front_speed = frame[vehicle_in_front].speed
            car_length = 4
            gap = dist - car_length
            acc_acceleration = self._acc.get_acceleration(self.MAX_SPEED, state.speed, in_front_speed, gap)
            acceleration = min(pid_acceleration, acc_acceleration)
        return acceleration

    def done(self, observation: Observation) -> bool:
        state = observation.frame[self.agent_id]
        ls = LineString(self.trajectory.path)
        p = Point(state.position)
        dist_along = ls.project(p)
        #lon_dist = p.distance(ls)
        dist_from_end = np.linalg.norm(state.position - self.trajectory.path[-1])
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

    def __stop_required(self, observation: Observation, target_wp_idx: int):
        ego_time_to_junction = self.trajectory.times[-1] - self.trajectory.times[target_wp_idx]
        times_to_junction = self._get_times_to_junction(observation.frame, observation.scenario_map,
                                                        ego_time_to_junction)
        time_until_clear = self._get_time_until_clear(ego_time_to_junction, times_to_junction)
        return time_until_clear > 0

    def next_action(self, observation: Observation) -> Action:
        state = observation.frame[self.agent_id]
        target_wp_idx, closest_idx = self.get_target_waypoint(state)
        target_waypoint = self.trajectory.path[target_wp_idx]
        close_to_junction_entry = len(self.trajectory.path) - target_wp_idx <= 4
        if close_to_junction_entry:
            if self.__stop_required(observation, target_wp_idx):
                target_velocity = 0
            else:
                target_velocity = 2
        else:
            target_velocity = self.trajectory.velocity[target_wp_idx]
        return self._get_action(target_waypoint, target_velocity, observation)


class CLManeuverFactory:
    maneuver_types = {"follow-lane": FollowLaneCL,
                      "switch-left": SwitchLaneLeftCL,
                      "switch-right": SwitchLaneRightCL,
                      "turn": TurnCL,
                      "give-way": GiveWayCL,
                      "trajectory": TrajectoryManeuverCL}

    @classmethod
    def create(cls, config: ManeuverConfig, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map):
        return cls.maneuver_types[config.type](config, agent_id, frame, scenario_map)