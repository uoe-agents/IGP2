import abc
from typing import Dict, List
from copy import copy
import numpy as np

from igp2.agent import AgentState
from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, FollowLane, ManeuverConfig, SwitchLaneLeft, SwitchLaneRight, SwitchLane
from igp2.trajectory import VelocityTrajectory


class MacroAction(abc.ABC):
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True):
        self.open_loop = open_loop
        self.agent_id = agent_id
        self.start_frame = frame
        self.scenario_map = scenario_map

        self._maneuvers = self.get_maneuvers()

    def done(self) -> bool:
        """ Returns True if the execution of the macro action has completed. """
        raise NotImplementedError

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Return True if the macro action is applicable in the given state of the environment. """
        raise NotImplementedError

    def get_trajectory(self) -> VelocityTrajectory:
        """ If open_loop is True then get the complete trajectory of the macro action.

        Returns:
            A VelocityTrajectory that describes the complete open loop trajectory of the macro action
        """
        if not self.open_loop:
            raise ValueError("Cannot get trajectory of closed-loop macro action!")
        if self._maneuvers is None:
            raise ValueError("Maneuver sequence of macro action was not initialised!")

        points = None
        velocity = None
        for maneuver in self._maneuvers:
            trajectory = maneuver.trajectory
            points = trajectory.path if points is None else np.append(points, trajectory.path, axis=0)
            velocity = trajectory.velocity if velocity is None else np.append(velocity, trajectory.velocity, axis=0)
        return VelocityTrajectory(points, velocity)

    def get_maneuvers(self) -> List[Maneuver]:
        """ Calculate the sequence of maneuvers for this MacroAction. """
        raise NotImplementedError

    @property
    def current_maneuver(self) -> Maneuver:
        """ The current maneuver being executed during closed loop control. """
        raise NotImplementedError

    @property
    def maneuvers(self):
        """ The complete maneuver sequence of the macro action. """
        return self._maneuvers


class Continue(MacroAction):
    def get_maneuvers(self) -> List[Maneuver]:
        if self.open_loop:
            current_lane = self.scenario_map.best_lane_at(self.start_frame[self.agent_id].position,
                                                          self.start_frame[self.agent_id].heading)
            endpoint = current_lane.midline.interpolate(1, normalized=True)
            config_dict = {
                "type": "follow-lane",
                "termination_point": np.array(endpoint.coords[0])
            }
            config = ManeuverConfig(config_dict)
            return [FollowLane(config, self.agent_id, self.start_frame, self.scenario_map)]

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        return FollowLane.applicable(state, scenario_map)


class ChangeLane(MacroAction):
    def __init__(self, left: bool, agent_id: int, frame: Dict[int, AgentState],
                 scenario_map: Map, open_loop: bool = True):
        self.left = left
        super(ChangeLane, self).__init__(agent_id, frame, scenario_map, open_loop)

    def get_maneuvers(self) -> List[Maneuver]:
        maneuvers = []
        state = self.start_frame[self.agent_id]
        current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
        current_distance = current_lane.distance_at(state.position)
        neighbour_lane_id = current_lane.id + (1 if np.sign(current_lane.id) > 0 else -1) * (-1 if self.left else 1)
        neighbour_lane = current_lane.lane_section.get_lane(neighbour_lane_id)

        if self.open_loop:
            t_change = SwitchLane.TARGET_SWITCH_LENGTH / state.speed

            # Get first time when lane change is possible
            oncoming_intervals = self._get_oncoming_vehicle_intervals(neighbour_lane)
            t_start = 0.0  # Count from time of start_frame
            for iv_start, iv_end in oncoming_intervals:
                if t_start < iv_end and iv_start < t_start + t_change:
                    t_start = iv_end

            # Check if we can complete lane change before hitting the end of the lane
            t_lane_end = (neighbour_lane.length - neighbour_lane.distance_at(state.position)) / state.speed
            assert t_start + t_change < t_lane_end, "Cannot finish lane change until end of current lane!"

            # Follow lane until lane is clear
            distance_until_change = t_start * Maneuver.MAX_SPEED
            lane_follow_end_distance = current_distance + distance_until_change
            lane_follow_end_point = state.position
            if t_start > 0.0:
                lane_follow_end_point = current_lane.point_at(lane_follow_end_distance)
                config_dict = {
                    "type": "follow-lane",
                    "termination_point": lane_follow_end_point
                }
                config = ManeuverConfig(config_dict)
                maneuvers.append(FollowLane(config, self.agent_id, self.start_frame, self.scenario_map))

            # Create switch lane maneuver
            config_dict = {
                "type": "switch-" + "left" if self.left else "right",
                "termination_point": neighbour_lane.point_at(
                    neighbour_lane.distance_at(lane_follow_end_point) + SwitchLane.TARGET_SWITCH_LENGTH)
            }
            config = ManeuverConfig(config_dict)
            new_frame = self._get_lane_change_frame(current_lane, lane_follow_end_point,
                                                    lane_follow_end_distance, t_start)
            if self.left:
                maneuvers.append(SwitchLaneLeft(config, self.agent_id, new_frame, self.scenario_map))
            else:
                maneuvers.append(SwitchLaneRight(config, self.agent_id, new_frame, self.scenario_map))
            return maneuvers

    def _get_oncoming_vehicle_intervals(self, neighbour_lane: Lane):
        oncoming_intervals = []
        state = self.start_frame[self.agent_id]
        for aid, agent in self.start_frame.items():
            if self.agent_id == aid:
                continue

            agent_lane = self.scenario_map.best_lane_at(agent.position, agent.heading)
            if agent_lane == neighbour_lane:
                d_speed = state.speed - agent.speed
                d_distance = neighbour_lane.distance_at(agent.position) - neighbour_lane.distance_at(state.position)

                # If heading in same direction and with same speed, then check if the distance allows for a lone change
                if np.isclose(d_speed, 0.0):
                    if np.abs(d_distance) < SwitchLane.MIN_SWITCH_LENGTH:
                        raise RuntimeError("Lane change is blocked by vehicle with same velocity in neighbouring lane.")
                    continue

                time_until_pass = d_distance / d_speed
                pass_time = np.abs(SwitchLane.MIN_SWITCH_LENGTH / d_speed)
                interval_end_time = time_until_pass + pass_time

                if interval_end_time > 0:
                    interval_start_time = max(0, time_until_pass - pass_time)
                    oncoming_intervals.append((interval_start_time, interval_end_time))

        oncoming_intervals = sorted(oncoming_intervals, key=lambda period: period[0])
        return oncoming_intervals

    def _get_lane_change_frame(self, current_lane, lane_follow_end_point, lane_follow_end_distance, t_start):
        new_frame = {}
        for aid, agent in self.start_frame.items():
            state = copy(agent)
            if aid == self.agent_id:
                state.position = lane_follow_end_point
                state.heading = current_lane.get_heading_at(lane_follow_end_distance)
                new_frame[aid] = state
            else:
                agent_lane = self.scenario_map.best_lane_at(agent.position, agent.heading)
                agent_distance = agent_lane.distance_at(agent.position) + t_start * agent.speed
                state.position = agent_lane.point_at(agent_distance)
                state.heading = agent_lane.get_heading_at(agent_distance)
        return new_frame


class ChangeLaneLeft(ChangeLane):
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True):
        super(ChangeLaneLeft, self).__init__(True, agent_id, frame, scenario_map, open_loop)

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        return SwitchLaneLeft.applicable(state, scenario_map)


class ChangeLaneRight(ChangeLane):
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True):
        super(ChangeLaneRight, self).__init__(False, agent_id, frame, scenario_map, open_loop)

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        return SwitchLaneRight.applicable(state, scenario_map)
