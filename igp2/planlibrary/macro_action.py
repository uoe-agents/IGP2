import abc
import logging
from typing import Dict, List, Optional, Type, Tuple
from copy import copy
import numpy as np
from shapely.geometry import Point

from igp2.agent import AgentState
from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, FollowLane, ManeuverConfig, SwitchLaneLeft, \
    SwitchLaneRight, SwitchLane, Turn, GiveWay
from igp2.trajectory import VelocityTrajectory
from igp2.util import all_subclasses

logger = logging.getLogger(__name__)


class MacroAction(abc.ABC):
    """ Base class for all MacroActions. """
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True,
                 **kwargs):
        """ Initialise a new MacroAction (MA)

        Args:
            agent_id: The ID of the agent, this MA is made for
            frame: The start state of the environment
            scenario_map: The road layout of the scenario
            open_loop: If True then use open loop control, else use closed loop control
        """
        self.open_loop = open_loop
        self.agent_id = agent_id
        self.start_frame = frame
        self.final_frame = None
        self.scenario_map = scenario_map

        self._maneuvers = self.get_maneuvers()

    @staticmethod
    def play_forward_macro_action(agent_id: int, scenario_map: Map,
                                  frame: Dict[int, AgentState], macro_action: "MacroAction"):
        """ Play forward current frame with the given macro action for the current agent.
        Assumes constant velocity lane follow behaviour for other agents.

        Args:
            agent_id: ID of the ego agent
            scenario_map: The road layout of the current scenario
            frame: The current frame of the environment
            macro_action: The macro action to play forward

        Returns:
            A new frame describing the future state of the environment
        """
        def _lane_at_distance(lane: Lane, ds: float) -> Tuple[Lane, float]:
            current_lane = lane
            total_length = 0.0
            while True:
                if total_length <= ds < total_length + current_lane.length:
                    return current_lane, ds - total_length
                total_length += current_lane.length
                successor = current_lane.link.successor
                if successor is None:
                    break
                current_lane = successor[0]
            logger.debug(f"No Lane found at distance {ds} for Agent ID{aid}!")
            return current_lane, current_lane.length

        if not macro_action:
            return frame

        trajectory = macro_action.get_trajectory()
        new_frame = {agent_id: trajectory.final_agent_state}
        duration = trajectory.duration
        for aid, agent in frame.items():
            if aid != agent_id:
                state = copy(agent)
                agent_lane = scenario_map.best_lane_at(agent.position, agent.heading)
                agent_distance = agent_lane.distance_at(agent.position) + duration * agent.speed
                final_lane, distance_in_lane = _lane_at_distance(agent_lane, agent_distance)
                state.position = final_lane.point_at(distance_in_lane)
                state.heading = final_lane.get_heading_at(distance_in_lane)
                new_frame[aid] = state
        return new_frame

    def get_maneuvers(self) -> List[Maneuver]:
        """ Calculate the sequence of maneuvers for this MacroAction. """
        raise NotImplementedError

    @property
    def maneuvers(self):
        """ The complete maneuver sequence of the macro action. """
        return self._maneuvers

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        """ Return True if the macro action is applicable in the given state of the environment. """
        raise NotImplementedError

    def done(self) -> bool:
        """ Returns True if the execution of the macro action has completed. """
        raise NotImplementedError

    @property
    def current_maneuver(self) -> Maneuver:
        """ The current maneuver being executed during closed loop control. """
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
            points = trajectory.path if points is None else np.append(points, trajectory.path[1:], axis=0)
            velocity = trajectory.velocity if velocity is None else np.append(velocity, trajectory.velocity[1:], axis=0)
        return VelocityTrajectory(points, velocity)

    @staticmethod
    def get_applicable_actions(agent_state: AgentState, scenario_map: Map) -> List[Type['MacroAction']]:
        """ Return all applicable macro actions.

        Args:
            agent_state: Current state of the examined agent
            scenario_map: The road layout of the scenario

        Returns:
            A list of applicable macro action types
        """
        actions = []
        for macro_action in all_subclasses(MacroAction):
            try:
                if macro_action.applicable(agent_state, scenario_map):
                    actions.append(macro_action)
            except NotImplementedError:
                continue
        return actions

    @staticmethod
    def get_possible_args(state: AgentState, scenario_map: Map, goal_point: np.ndarray = None) -> List[Dict]:
        """ Return a list of keyword arguments used to initialise all possible variations of a macro action.
        Currently, only Exit returns more than one option, giving the Exits to all possible leaving points.

        Args:
            state: Current state of the agent
            scenario_map: The road layout of the scenario
            goal_point: Optional goal point to use during AStar planning

        Returns:
            A list of possible initialisations in the current state
        """
        return [{}]


class Continue(MacroAction):
    """ Follow the current lane until the given point or to the end of the lane. """
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True,
                 termination_point: Point = None):
        """ Initialise a new Continue MA.

        Args:
            termination_point: The optional point of termination
        """
        self.termination_point = termination_point
        super().__init__(agent_id, frame, scenario_map, open_loop)

    def get_maneuvers(self) -> List[Maneuver]:
        state = self.start_frame[self.agent_id]
        maneuvers = []
        if self.open_loop:
            current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
            endpoint = self.termination_point
            if endpoint is None:
                endpoint = current_lane.midline.interpolate(1, normalized=True)
            config_dict = {
                "type": "follow-lane",
                "termination_point": np.array(endpoint.coords[0])
            }
            config = ManeuverConfig(config_dict)
            maneuvers = [FollowLane(config, self.agent_id, self.start_frame, self.scenario_map)]
            self.final_frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map,
                                                              self.start_frame, maneuvers[-1])
        return maneuvers

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        in_junction = scenario_map.best_lane_at(state.position, state.heading).parent_road.junction is not None
        return (FollowLane.applicable(state, scenario_map) and not in_junction and
                not GiveWay.applicable(state, scenario_map))

    @staticmethod
    def get_possible_args(state: AgentState, scenario_map: Map, goal_point: np.ndarray = None) -> List[Dict]:
        if goal_point is not None:
            current_lane = scenario_map.best_lane_at(state.position, state.heading)
            goal_lanes = scenario_map.lanes_at(goal_point)
            if current_lane in goal_lanes:
                return [{"termination_point": Point(goal_point)}]
        return [{}]


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
            frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, maneuvers[-1])

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
            self.final_frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, maneuvers[-1])
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
        neighbour_lane = current_lane.lane_section.get_lane(neighbour_lane_id)  # TODO: Deal with change lanes over a sequence of target lanes

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
                frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)

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
            self.final_frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, maneuvers[-1])
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
    LANE_ANGLE_THRESHOLD = np.pi / 9  # The maximum angular distance between the current heading the heading of a lane
    TURN_TARGET_THRESHOLD = 1  # Threshold for checking if turn target is within distance of another point

    def __init__(self, turn_target: np.ndarray, agent_id: int, frame: Dict[int, AgentState],
                 scenario_map: Map, open_loop: bool = True):
        self.turn_target = turn_target
        super(Exit, self).__init__(agent_id, frame, scenario_map, open_loop)

    def get_maneuvers(self) -> List[Maneuver]:
        maneuvers = []
        state = self.start_frame[self.agent_id]
        in_junction = self.scenario_map.junction_at(state.position) is not None
        current_lane = self._find_current_lane(state, in_junction)
        current_distance = current_lane.distance_at(state.position)

        if self.open_loop:
            frame = self.start_frame

            connecting_lane = current_lane
            if not in_junction:
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
                    frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)

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
                frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)

            # Add turn
            config_dict = {
                "type": "turn",
                "termination_point": self.turn_target,
                "junction_road_id": connecting_lane.parent_road.id,
                "junction_lane_id": connecting_lane.id
            }
            config = ManeuverConfig(config_dict)
            maneuvers.append(Turn(config, self.agent_id, frame, self.scenario_map))
            self.final_frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, maneuvers[-1])

        return maneuvers

    def _nearest_lane_to_goal(self, lane_list: List[Lane]) -> Lane:
        best_lane = None
        best_distance = np.inf
        for connecting_lane in lane_list:
            distance = np.linalg.norm(self.turn_target - np.array(connecting_lane.midline.coords[-1]))
            if distance < self.TURN_TARGET_THRESHOLD and distance < best_distance:
                best_lane = connecting_lane
        return best_lane

    def _find_current_lane(self, state: AgentState, in_junction: bool) -> Lane:
        if not in_junction:
            return self.scenario_map.best_lane_at(state.position, state.heading)
        all_lanes = self.scenario_map.lanes_at(state.position)
        return self._nearest_lane_to_goal(all_lanes)

    def _find_connecting_lane(self, current_lane: Lane) -> Optional[Lane]:
        return self._nearest_lane_to_goal(current_lane.link.successor)

    @staticmethod
    def applicable(state: AgentState, scenario_map: Map) -> bool:
        in_junction = scenario_map.junction_at(state.position) is not None
        if ContinueNextExit.applicable(state, scenario_map):
            return False
        if in_junction:
            return Turn.applicable(state, scenario_map)
        else:
            return GiveWay.applicable(state, scenario_map)

    @staticmethod
    def get_possible_args(state: AgentState, scenario_map: Map, goal_point: np.ndarray = None) -> List[Dict]:
        ret = []
        in_junction = scenario_map.junction_at(state.position) is not None

        if in_junction:
            lanes_at = scenario_map.lanes_at(state.position)
            if len(lanes_at) == 1:
                lanes = lanes_at
            else:
                lanes = scenario_map.lanes_within_angle(state.position, state.heading, Exit.LANE_ANGLE_THRESHOLD)

            for lane in lanes:
                new_dict = {"turn_target": np.array(lane.midline.coords[-1])}
                ret.append(new_dict)
        else:
            current_lane = scenario_map.best_lane_at(state.position, state.heading)
            for connecting_lane in current_lane.link.successor:
                if not scenario_map.road_in_roundabout(connecting_lane.parent_road):
                    new_dict = {"turn_target": np.array(connecting_lane.midline.coords[-1])}
                    ret.append(new_dict)

        return ret
