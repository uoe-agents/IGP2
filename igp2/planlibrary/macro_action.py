import abc
from typing import Dict, List, Optional
from copy import copy
import numpy as np

from igp2.agent import AgentState
from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, FollowLane, ManeuverConfig, SwitchLaneLeft, SwitchLaneRight, SwitchLane, \
    Turn, GiveWay
from igp2.trajectory import VelocityTrajectory


class MacroAction(abc.ABC):
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True):
        self.open_loop = open_loop
        self.agent_id = agent_id
        self.start_frame = frame
        self.final_frame = None
        self.scenario_map = scenario_map

        self._maneuvers = self.get_maneuvers()

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Return True if the macro action is applicable in the given state of the environment. """
        raise NotImplementedError

    # def done(self) -> bool:
    #     """ Returns True if the execution of the macro action has completed. """
    #     raise NotImplementedError
    #
    # @property
    # def current_maneuver(self) -> Maneuver:
    #     """ The current maneuver being executed during closed loop control. """
    #     raise NotImplementedError

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

    def play_forward_maneuver(self, frame: Dict[int, AgentState], maneuver: Maneuver) -> Dict[int, AgentState]:
        if not maneuver:
            return frame

        new_frame = {self.agent_id: frame[self.agent_id]}
        duration = maneuver.trajectory.duration
        for aid, agent in frame.items():
            state = copy(agent)
            if aid == self.agent_id:
                if len(maneuver.trajectory.path) > 1:
                    diff = maneuver.trajectory.path[-1] - maneuver.trajectory.path[-2]
                    current_lane = self.scenario_map.best_lane_at(
                        maneuver.trajectory.path[-1],
                        np.arctan2(diff[1], diff[0])
                    )
                else:
                    current_lane = self.scenario_map.best_lane_at(
                        maneuver.trajectory.path[-1]
                    )
                state.position = maneuver.trajectory.path[-1]
                state.heading = current_lane.get_heading_at(current_lane.distance_at(maneuver.trajectory.path[-1]))
            else:
                agent_lane = self.scenario_map.best_lane_at(agent.position, agent.heading)
                agent_distance = agent_lane.distance_at(agent.position) + duration * agent.speed
                state.position = agent_lane.point_at(agent_distance)
                state.heading = agent_lane.get_heading_at(agent_distance)
            new_frame[aid] = state
        return new_frame

    def get_maneuvers(self) -> List[Maneuver]:
        """ Calculate the sequence of maneuvers for this MacroAction. """
        raise NotImplementedError

    @property
    def maneuvers(self):
        """ The complete maneuver sequence of the macro action. """
        return self._maneuvers


class Continue(MacroAction):
    def get_maneuvers(self) -> List[Maneuver]:
        state = self.start_frame[self.agent_id]
        maneuvers = []
        if self.open_loop:
            current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
            endpoint = current_lane.midline.interpolate(1, normalized=True)
            config_dict = {
                "type": "follow-lane",
                "termination_point": np.array(endpoint.coords[0])
            }
            config = ManeuverConfig(config_dict)
            maneuvers = [FollowLane(config, self.agent_id, self.start_frame, self.scenario_map)]
            self.final_frame = self.play_forward_maneuver(self.start_frame, maneuvers[-1])
        return maneuvers

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        return FollowLane.applicable(state, scenario_map)


class ContinueNextExit(MacroAction):
    """ Continue in the non-outer lane of a roundabout until after the next junction. """
    def get_maneuvers(self) -> List[Maneuver]:
        state = self.start_frame[self.agent_id]
        maneuvers = []
        if self.open_loop:
            # First go till end of current lane
            frame = self.start_frame
            current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
            endpoint = current_lane.midline.interpolate(1, normalized=True)
            config_dict = {
                "type": "follow-lane",
                "termination_point": np.array(endpoint.coords[0])
            }
            config = ManeuverConfig(config_dict)
            man = FollowLane(config, self.agent_id, frame, self.scenario_map)
            maneuvers.append(man)
            frame = self.play_forward_maneuver(frame, maneuvers[-1])

            # Then go straight through the junction
            turn_lane = current_lane.link.successor[0]
            endpoint = turn_lane.midline.interpolate(1, normalized=True)
            config_dict = {
                "type": "turn",
                "termination_point": np.array(endpoint.coords[0]),
                "junction_lane_id": turn_lane.id,
                "junction_road_id": turn_lane.parent_road.id
            }
            config = ManeuverConfig(config_dict)
            man = Turn(config, self.agent_id, frame, self.scenario_map)
            maneuvers.append(man)
            self.final_frame = self.play_forward_maneuver(frame, maneuvers[-1])
        return maneuvers

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        all_lane_ids = [lane.id for lane in current_lane.lane_section.all_lanes if lane != current_lane]
        return (scenario_map.in_roundabout(state.position, state.heading) and
                current_lane.parent_road.junction is None and  # TODO: Assume cannot continue to next exit while going through junction
                not all([np.abs(current_lane.id) > np.abs(lid) for lid in all_lane_ids]) and  # Not in outer lane
                FollowLane.applicable(state, scenario_map))


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
            frame = self.start_frame
            d_change = SwitchLane.TARGET_SWITCH_LENGTH
            oncoming_intervals = self._get_oncoming_vehicle_intervals(neighbour_lane)
            t_lane_end = (neighbour_lane.length - neighbour_lane.distance_at(state.position)) / state.speed

            # Get first time when lane change is possible
            while d_change >= SwitchLane.MIN_SWITCH_LENGTH:
                t_change = d_change / state.speed
                t_start = 0.0  # Count from time of start_frame
                for iv_start, iv_end in oncoming_intervals:
                    if t_start < iv_end and iv_start < t_start + t_change:
                        t_start = iv_end

                if t_start + t_change >= t_lane_end:
                    d_change -= 5  # Try lane change with shorter length
                else:
                    break
            else:
                assert False, "Cannot finish lane change until end of current lane!"

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
                man = FollowLane(config, self.agent_id, frame, self.scenario_map)
                maneuvers.append(man)
                frame = self.play_forward_maneuver(frame, man)

            # Create switch lane maneuver
            config_dict = {
                "type": "switch-" + "left" if self.left else "right",
                "termination_point": neighbour_lane.point_at(
                    neighbour_lane.distance_at(lane_follow_end_point) + d_change)
            }
            config = ManeuverConfig(config_dict)
            if self.left:
                maneuvers.append(SwitchLaneLeft(config, self.agent_id, frame, self.scenario_map))
            else:
                maneuvers.append(SwitchLaneRight(config, self.agent_id, frame, self.scenario_map))
            self.final_frame = self.play_forward_maneuver(frame, maneuvers[-1])
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


class Exit(MacroAction):
    GIVE_WAY_DISTANCE = 10  # Begin give-way if closer than this value to the junction
    TURN_TARGET_THRESHOLD = 1  # Threshold for checking if turn target is within distance of another point

    def __init__(self, turn_target: np.ndarray, agent_id: int, frame: Dict[int, AgentState],
                 scenario_map: Map, open_loop: bool = True):
        self.turn_target = turn_target
        super(Exit, self).__init__(agent_id, frame, scenario_map, open_loop)

    def get_maneuvers(self) -> List[Maneuver]:
        maneuvers = []
        state = self.start_frame[self.agent_id]
        current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
        current_distance = current_lane.distance_at(state.position)

        if self.open_loop:
            frame = self.start_frame

            # Follow lane until start of turn if outside of give-way distance
            if current_lane.length - current_distance > self.GIVE_WAY_DISTANCE:
                distance_of_termination = current_lane.length - self.GIVE_WAY_DISTANCE
                lane_follow_termination = current_lane.point_at(distance_of_termination)
                config_dict = {
                    "type": "follow-lane",
                    "termination_point": lane_follow_termination
                }
                config = ManeuverConfig(config_dict)
                man = FollowLane(config, self.agent_id, frame, self.scenario_map)
                maneuvers.append(man)
                frame = self.play_forward_maneuver(frame, man)

            connecting_lane = self._find_connecting_lane(current_lane)

            # Add give-way maneuver
            config_dict = {
                "type": "give-way",
                "termination_point": current_lane.midline.coords[-1],
                "junction_road_id": connecting_lane.parent_road.id,
                "junction_lane_id": connecting_lane.id
            }
            config = ManeuverConfig(config_dict)
            man = GiveWay(config, self.agent_id, frame, self.scenario_map)
            maneuvers.append(man)
            frame = self.play_forward_maneuver(frame, man)

            # Add turn
            config_dict = {
                "type": "turn",
                "termination_point": self.turn_target,
                "junction_road_id": connecting_lane.parent_road.id,
                "junction_lane_id": connecting_lane.id
            }
            config = ManeuverConfig(config_dict)
            maneuvers.append(Turn(config, self.agent_id, frame, self.scenario_map))
            self.final_frame = self.play_forward_maneuver(frame, maneuvers[-1])

        return maneuvers

    def _find_connecting_lane(self, current_lane: Lane) -> Optional[Lane]:
        best_lane = None
        best_distance = np.inf
        for connecting_lane in current_lane.link.successor:
            distance = np.linalg.norm(self.turn_target - np.array(connecting_lane.midline.coords[-1]))
            if distance < self.TURN_TARGET_THRESHOLD and distance < best_distance:
                best_lane = connecting_lane
        return best_lane

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        return Turn.applicable(state, scenario_map)
