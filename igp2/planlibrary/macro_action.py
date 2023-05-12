import igp2 as ip
import abc
import logging
import numpy as np
from typing import Dict, List, Optional, Type, Tuple
from copy import copy
from shapely.geometry import Point, LineString

from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig

logger = logging.getLogger(__name__)


class MacroAction(abc.ABC):
    """ Base class for all MacroActions. """

    def __init__(self, agent_id: int, frame: Dict[int, ip.AgentState], scenario_map: ip.Map, open_loop: bool = True,
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
        self._current_maneuver = None
        self._current_maneuver_id = 0
        if not self.open_loop:
            self._advance_maneuver(ip.Observation(frame, scenario_map))

    def __repr__(self):
        return self.__class__.__name__

    @staticmethod
    def play_forward_macro_action(agent_id: int, scenario_map: ip.Map,
                                  frame: Dict[int, ip.AgentState], macro_action: "MacroAction"):
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

        def _lane_at_distance(lane: ip.Lane, ds: float) -> Tuple[Optional[ip.Lane], float]:
            current_lane = lane
            total_length = 0.0
            while True:
                if total_length <= ds < total_length + current_lane.length:
                    return current_lane, ds - total_length
                total_length += current_lane.length
                successor = current_lane.link.successor
                if successor is None:
                    break
                if len(successor) > 1:
                    return current_lane, current_lane.length
                current_lane = successor[0]
            return None, current_lane.length

        if not macro_action:
            return frame

        trajectory = macro_action.get_trajectory()
        new_frame = {agent_id: trajectory.final_agent_state}
        duration = trajectory.duration
        for aid, agent in frame.items():
            if aid != agent_id:
                state = copy(agent)
                agent_lane = scenario_map.best_lane_at(agent.position, agent.heading)
                if agent_lane is None:
                    continue
                agent_distance = agent_lane.distance_at(agent.position) + duration * agent.speed
                final_lane, distance_in_lane = _lane_at_distance(agent_lane, agent_distance)
                if final_lane is None:
                    continue
                state.position = final_lane.point_at(distance_in_lane)
                state.heading = final_lane.get_heading_at(distance_in_lane)
                new_frame[aid] = state
        return new_frame

    def get_maneuvers(self) -> List[Maneuver]:
        """ Calculate the sequence of maneuvers for this MacroAction. """
        raise NotImplementedError

    @staticmethod
    def applicable(state: ip.AgentState, scenario_map: ip.Map) -> bool:
        """ Return True if the macro action is applicable in the given state of the environment. """
        raise NotImplementedError

    def reset(self):
        """ Reset the internal state of closed-loop macro actions. """
        if not self.open_loop:
            self._current_maneuver = None
            self._current_maneuver_id = 0

    def done(self, observation: ip.Observation) -> bool:
        """ Returns True if the execution of the macro action has completed. """
        return self._current_maneuver_id + 1 >= len(self._maneuvers) and self._current_maneuver.done(observation)

    def next_action(self, observation: ip.Observation) -> Optional[ip.Action]:
        """ Return the next action of a closed-loop macro action given by its current maneuver. If the current
        maneuver is done, then advance to the next maneuver. """
        self._advance_maneuver(observation)
        return self.current_maneuver.next_action(observation)

    def _advance_maneuver(self, observation: ip.Observation):
        if not self._maneuvers:
            raise RuntimeError("Macro action has no maneuvers.")
        else:
            if self._current_maneuver is None:
                self._current_maneuver = self._maneuvers[self._current_maneuver_id]
            elif self._current_maneuver.done(observation):
                self._current_maneuver_id += 1
                if self._current_maneuver_id >= len(self._maneuvers):
                    raise RuntimeError("No more maneuvers to execute in macro action.")
                else:
                    self._current_maneuver = self._maneuvers[self._current_maneuver_id]

    def get_trajectory(self) -> ip.VelocityTrajectory:
        """ If open_loop is True then get the complete trajectory of the macro action.

        Returns:
            A VelocityTrajectory that describes the complete open loop trajectory of the macro action
        """
        if self._maneuvers is None:
            raise ValueError("Maneuver sequence of macro action was not initialised!")

        points = None
        velocity = None
        for maneuver in self._maneuvers:
            trajectory = maneuver.trajectory
            points = trajectory.path if points is None else \
                np.append(points, trajectory.path[1:], axis=0)
            velocity = trajectory.velocity if velocity is None else \
                np.append(velocity, trajectory.velocity[1:], axis=0)
        # delete repeated points, car shall not be static in roundabouts
        idx = np.where(np.abs(np.diff(points[:, 0])) + np.abs(np.diff(points[:, 1])) > 0)
        if len(points) - len(idx[0]) > 1:
            points = np.r_[points[idx], points[-1].reshape(1, 2)]
            velocity = np.r_[velocity[idx], velocity[-1]]
        return ip.VelocityTrajectory(points, velocity)

    def to_closed_loop(self):
        """ Convert an open-loop macro action to closed-loop.
        If already closed-loop then this will reset the macro action's state.
        """
        if self.open_loop:
            mans = []
            for i, man in enumerate(self._maneuvers):
                mans.append(ip.CLManeuverFactory.create(
                    man.config, man.agent_id, man.frame, self.scenario_map))
            self._maneuvers = mans
            self.open_loop = False
        if not self.open_loop:
            self.reset()
            self._advance_maneuver(ip.Observation(self.start_frame, self.scenario_map))

    @staticmethod
    def get_applicable_actions(agent_state: ip.AgentState, scenario_map: ip.Map, goal: ip.Goal = None) \
            -> List[Type['MacroAction']]:
        """ Return all applicable macro actions.

        Args:
            agent_state: Current state of the examined agent
            scenario_map: The road layout of the scenario
            goal: If given and ahead within current lane boundary, then will always return at least a Continue

        Returns:
            A list of applicable macro action types
        """
        actions = []

        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        if goal is not None:
            goal_point = goal.point_on_lane(current_lane)
            if current_lane.boundary.contains(goal_point) and \
                    current_lane.distance_at(agent_state.position) < current_lane.distance_at(goal_point):
                actions = [Continue]

        all_subclasses = ip.util.all_subclasses(MacroAction)
        all_subclasses.discard(ChangeLane) # since there is no applicability in this class
        for macro_action in all_subclasses:
            try:
                if macro_action not in actions and macro_action.applicable(agent_state, scenario_map):
                    actions.append(macro_action)
            except NotImplementedError:
                continue
        return actions

    @staticmethod
    def get_possible_args(state: ip.AgentState, scenario_map: ip.Map, goal: ip.Goal = None) -> List[Dict]:
        """ Return a list of keyword arguments used to initialise all possible variations of a macro action.
        Currently, only Exit returns more than one option, giving the Exits to all possible leaving points.

        Args:
            state: Current state of the agent
            scenario_map: The road layout of the scenario
            goal: Optional goal to use during AStar planning

        Returns:
            A list of possible initialisations in the current state
        """
        return [{}]

    @property
    def maneuvers(self) -> List[Maneuver]:
        """ The complete maneuver sequence of the macro action. """
        return self._maneuvers

    @property
    def current_maneuver(self) -> Maneuver:
        """ The current maneuver being executed during closed loop control. """
        return self._current_maneuver


class Continue(MacroAction):
    """ Follow the current lane until the given point or to the end of the lane.
     If the current lane is split across multiple roads then follow lane until a junction is encountered. """

    def __init__(self, agent_id: int, frame: Dict[int, ip.AgentState], scenario_map: ip.Map, open_loop: bool = True,
                 termination_point: Point = None):
        """ Initialise a new Continue MA.

        Args:
            termination_point: The optional point of termination
        """
        self.termination_point = termination_point
        super().__init__(agent_id, frame, scenario_map, open_loop)

    def __repr__(self):
        termination = np.round(np.array(self.termination_point), 3) \
            if self.termination_point is not None else ''
        return f"Continue({termination})"

    def get_maneuvers(self) -> List[Maneuver]:
        state = self.start_frame[self.agent_id]
        current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
        endpoint = self.termination_point

        configs = []
        if endpoint is not None:
            config_dict = {"type": "follow-lane", "termination_point": endpoint}
            configs.append(config_dict)
        else:
            lane = current_lane
            while lane is not None:
                endpoint = lane.midline.interpolate(1, normalized=True)
                config_dict = {"type": "follow-lane", "termination_point": np.array(endpoint.coords[0])}
                configs.append(config_dict)
                in_roundabout = self.scenario_map.road_in_roundabout(lane.parent_road)
                succ = lane.link.successor
                lane = None
                if succ is not None:
                    if any([s.parent_road.junction is not None for s in succ]):
                        if not in_roundabout:
                            configs.pop()  # Last config lead to a junction not in a roundabout so remove it
                    elif len(succ) == 1 and succ[0] != current_lane:
                        lane = succ[0]

        maneuvers = []
        current_frame = self.start_frame
        for config_dict in configs:
            config = ManeuverConfig(config_dict)
            if self.open_loop:
                man = ip.FollowLane(config, self.agent_id, current_frame, self.scenario_map)
            else:
                man = ip.CLManeuverFactory.create(config, self.agent_id, current_frame, self.scenario_map)
            maneuvers.append(man)
            current_frame = Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map,
                                                           current_frame, maneuvers[-1], 0.1)
        self.final_frame = current_frame
        return maneuvers

    @staticmethod
    def applicable(state: ip.AgentState, scenario_map: ip.Map) -> bool:
        """ True if vehicle on a lane, and not approaching junction or not in junction"""
        current_road = scenario_map.best_road_at(state.position, state.heading)
        in_junction = current_road.junction is not None
        in_roundabout = scenario_map.road_in_roundabout(current_road)
        return (ip.FollowLane.applicable(state, scenario_map) and
                not in_junction and
                (not Exit.applicable(state, scenario_map) or in_roundabout))

    @staticmethod
    def get_possible_args(state: ip.AgentState, scenario_map: ip.Map, goal: ip.Goal = None) -> List[Dict]:
        """ Return empty dictionary if no goal point is provided, otherwise check if goal point in lane and
        return center of goal point as termination point. """
        if goal is not None:
            current_lane = scenario_map.best_lane_at(state.position, state.heading)
            gp = goal.point_on_lane(current_lane)
            if current_lane.boundary.contains(gp):
                return [{"termination_point": np.array(gp.coords)[0]}]
        return [{}]


class ChangeLane(MacroAction):
    CHECK_ONCOMING = False

    def __init__(self, target_sequence: List[ip.Lane], left: bool, agent_id: int, frame: Dict[int, ip.AgentState],
                 scenario_map: ip.Map, open_loop: bool = True):
        self.target_sequence = target_sequence
        self.left = left
        super(ChangeLane, self).__init__(agent_id, frame, scenario_map, open_loop)

    def __repr__(self):
        lane_seq_str = "->".join([f"[{lane.parent_road.id}:{lane.id}]" for lane in self.target_sequence])
        return f"ChangeLane{'Left' if self.left else 'Right'}({lane_seq_str})"

    def get_maneuvers(self) -> List[Maneuver]:
        maneuvers = []
        state = self.start_frame[self.agent_id]
        current_lane = self.scenario_map.best_lane_at(state.position, state.heading)
        current_distance = current_lane.distance_at(state.position)
        target_midline = ip.Maneuver.get_lane_path_midline(self.target_sequence)

        frame = self.start_frame
        d_lane_end = target_midline.length - target_midline.project(Point(state.position))
        d_change = max(ip.SwitchLane.MIN_SWITCH_LENGTH, min(ip.SwitchLane.TARGET_SWITCH_LENGTH, d_lane_end))
        lane_follow_end_point = state.position

        assert d_lane_end > ip.SwitchLane.MIN_SWITCH_LENGTH, "Cannot finish lange change within given lanes."

        # Check for oncoming vehicles and free sections in target lane if flag is set
        if ChangeLane.CHECK_ONCOMING:
            oncoming_intervals = self._get_oncoming_vehicle_intervals(self.target_sequence, target_midline)
            t_lane_end = d_lane_end / state.speed

            # Get first time when lane change is possible
            while d_change >= ip.SwitchLane.MIN_SWITCH_LENGTH:
                t_change = d_change / state.speed
                t_start = 0.0  # Count from time of start_frame
                for iv_start, iv_end, d_distance in oncoming_intervals:
                    if np.abs(d_distance) < d_change and t_start < iv_end and iv_start < t_start + t_change:
                        t_start = iv_end

                if t_start + t_change >= t_lane_end:
                    d_change -= 5  # Try lane change with shorter length
                else:
                    break
            else:
                raise RuntimeError("Cannot finish lane change until end of current lane!")

            # Follow lane until target lane is clear
            distance_until_change = t_start * state.speed  # Maneuver.MAX_SPEED
            lane_follow_end_distance = current_distance + distance_until_change
            if t_start > 0.0:
                lane_follow_end_point = current_lane.point_at(lane_follow_end_distance)
                config_dict = {
                    "type": "follow-lane",
                    "termination_point": lane_follow_end_point
                }
                config = ip.ManeuverConfig(config_dict)
                if self.open_loop:
                    man = ip.FollowLane(config, self.agent_id, frame, self.scenario_map)
                else:
                    man = ip.CLManeuverFactory.create(config, self.agent_id, frame, self.scenario_map)
                maneuvers.append(man)
                frame = ip.Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)

        # Create switch lane maneuver
        config_dict = {
            "type": "switch-" + ("left" if self.left else "right"),
            "termination_point": target_midline.interpolate(
                target_midline.project(Point(lane_follow_end_point)) + d_change),
            "lane_sequence": self.target_sequence
        }
        config = ip.ManeuverConfig(config_dict)
        if self.open_loop:
            if self.left:
                man = ip.SwitchLaneLeft(config, self.agent_id, frame, self.scenario_map)
            else:
                man = ip.SwitchLaneRight(config, self.agent_id, frame, self.scenario_map)
        else:
            man = ip.CLManeuverFactory.create(config, self.agent_id, frame, self.scenario_map)
        maneuvers.append(man)
        self.final_frame = ip.Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)
        return maneuvers

    @staticmethod
    def check_applicability(state: ip.AgentState, scenario_map: ip.Map, left: bool) -> bool:
        """ True if current lane not in junction, or at appropriate distance from a junction """
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        ds = current_lane.distance_at(state.position)
        in_junction = current_lane.parent_road.junction is not None
        in_roundabout = scenario_map.road_in_roundabout(current_lane.parent_road)

        # Calculate distance to next junction while lane change is possible
        if in_junction:
            lane = current_lane.link.successor[0]
        else:
            lane = current_lane

        dist_to_next_junction = current_lane.length - ds if in_junction else -ds
        while lane is not None and lane.parent_road.junction is None:
            if len(scenario_map.get_adjacent_lanes(lane)) < 1:
                break
            dist_to_next_junction += lane.length
            lane = lane.link.successor[0] if lane.link.successor is not None else None

        # Allow lane change in roundabouts if next roundabout junction is far enough
        if in_roundabout:
            return left or dist_to_next_junction > ip.SwitchLane.MIN_SWITCH_LENGTH

        # Otherwise disallow lane change if in a junction or not far enough
        return not in_junction and \
            dist_to_next_junction > ip.SwitchLane.TARGET_SWITCH_LENGTH + state.metadata.length

    def _get_oncoming_vehicle_intervals(self, target_lane_sequence: List[ip.Lane], target_midline: LineString):
        oncoming_intervals = []
        state = self.start_frame[self.agent_id]
        for aid, agent in self.start_frame.items():
            if self.agent_id == aid:
                continue

            agent_lanes = self.scenario_map.lanes_at(agent.position)
            if any([ll in target_lane_sequence for ll in agent_lanes]):
                d_speed = state.speed - agent.speed
                d_distance = target_midline.project(Point(agent.position)) - \
                    target_midline.project(Point(state.position))

                # If heading in same direction and with same speed, then check if the distance allows for a lone change
                if np.isclose(d_speed, 0.0):
                    if np.abs(d_distance) < ip.SwitchLane.MIN_SWITCH_LENGTH:
                        raise RuntimeError("Lane change is blocked by vehicle with "
                                           "same velocity in neighbouring lane.")
                    continue

                time_until_pass = d_distance / d_speed
                pass_time = np.abs(ip.SwitchLane.MIN_SWITCH_LENGTH / d_speed)
                interval_end_time = time_until_pass + pass_time

                if interval_end_time > 0:
                    interval_start_time = max(0, time_until_pass - pass_time)
                    oncoming_intervals.append((interval_start_time, interval_end_time, d_distance))

        oncoming_intervals = sorted(oncoming_intervals, key=lambda period: period[0])
        return oncoming_intervals

    @staticmethod
    def get_target_lane(current_lane: ip.Lane, left: bool) -> ip.Lane:
        # Get target lane based on direction, lane change side and current lane ID
        tid = current_lane.id + (1 if np.sign(current_lane.id) > 0 else -1) * (-1 if left else 1)
        return current_lane.lane_section.get_lane(tid)

    @staticmethod
    def get_possible_lanes(state: ip.AgentState, scenario_map: ip.Map,
                           goal: ip.Goal = None, left: bool = True) -> List[List[ip.Lane]]:
        """ Returns all possible lane changes when passing through a junction and there are multiple valid
        lane sequences. This will only really be applied when the vehicle is passing through a roundabout. """
        ls = [[]]
        current_lane = scenario_map.best_lane_at(state.position, state.heading)
        distance = -current_lane.distance_at(state.position)
        while current_lane is not None and distance <= ip.SwitchLane.TARGET_SWITCH_LENGTH:
            target_lane = ChangeLane.get_target_lane(current_lane, left)
            if target_lane is None or target_lane.id == 0:
                break

            # If in a roundabout and passing through a junction only select the exit road if available.
            #  Passing through the outer lane in a roundabout is not allowed so no point selecting that road.
            in_roundabout = scenario_map.road_in_roundabout(current_lane.parent_road)
            in_junction = current_lane.parent_road.junction is not None
            if distance > 0 and in_junction and in_roundabout:
                junction_lanes = scenario_map.lanes_at(target_lane.point_at(0.01))
                for lane in junction_lanes:
                    if not scenario_map.road_in_roundabout(lane.parent_road):
                        target_lane = lane
                        current_lane = None
                        break

            ls[0].append(target_lane)
            distance += target_lane.length

            if current_lane is None:
                break

            # Find successor lane to continue iteration
            successors = current_lane.link.successor
            current_lane = None
            if successors is not None:
                has_adjacent_lanes = [ln for ln in successors if len(scenario_map.get_adjacent_lanes(ln)) > 0]
                if len(has_adjacent_lanes) == 1:
                    current_lane = has_adjacent_lanes[0]
        return ls


class ChangeLaneLeft(ChangeLane):
    def __init__(self, target_sequence: List[ip.Lane],
                 agent_id: int, frame: Dict[int, ip.AgentState], scenario_map: ip.Map, open_loop: bool = True):
        super(ChangeLaneLeft, self).__init__(target_sequence, True, agent_id, frame, scenario_map, open_loop)

    @staticmethod
    def applicable(state: ip.AgentState, scenario_map: ip.Map) -> bool:
        """ True if valid target lane on the left and lane change is valid. """
        return ip.SwitchLaneLeft.applicable(state, scenario_map) and \
            ChangeLane.check_applicability(state, scenario_map, True)

    @staticmethod
    def get_possible_args(state: ip.AgentState, scenario_map: ip.Map, goal: ip.Goal = None) -> List[Dict]:
        ls = ChangeLane.get_possible_lanes(state, scenario_map, goal, True)
        return [{"target_sequence": s} for s in ls]


class ChangeLaneRight(ChangeLane):
    def __init__(self, target_sequence: List[ip.Lane],
                 agent_id: int, frame: Dict[int, ip.AgentState], scenario_map: ip.Map, open_loop: bool = True):
        super(ChangeLaneRight, self).__init__(target_sequence, False, agent_id, frame, scenario_map, open_loop)

    @staticmethod
    def applicable(state: ip.AgentState, scenario_map: ip.Map) -> bool:
        """ True if valid target lane on the right and lane change is valid. """
        return ip.SwitchLaneRight.applicable(state, scenario_map) and \
            ChangeLane.check_applicability(state, scenario_map, False)

    @staticmethod
    def get_possible_args(state: ip.AgentState, scenario_map: ip.Map, goal: ip.Goal = None) -> List[Dict]:
        ls = ChangeLane.get_possible_lanes(state, scenario_map, goal, False)
        return [{"target_sequence": s} for s in ls]


class Exit(MacroAction):
    TURN_TARGET_THRESHOLD = 1  # Threshold for checking if turn target is within distance of another point

    def __init__(self, turn_target: np.ndarray, agent_id: int, frame: Dict[int, ip.AgentState],
                 scenario_map: ip.Map, open_loop: bool = True, stop: bool = True):
        self.turn_target = turn_target
        self.stop = stop
        super(Exit, self).__init__(agent_id, frame, scenario_map, open_loop)

        # Calculate the orientation of the turn. If the returned value is less than 0 then the turn is clockwise (right)
        #  If it is larger than 0 it is oriented counter-clockwise (left).
        #  If it is zero, the turn is a straight line.
        eps = 0.1
        trajectory = self.maneuvers[-1].trajectory
        mean_angular_vel = np.dot(trajectory.timesteps,
                                  trajectory.angular_velocity)
        if mean_angular_vel < -eps: self.orientation = -1
        elif mean_angular_vel > eps: self.orientation = 1
        else: self.orientation = 0

    def __repr__(self):
        direction = "left" if self.orientation > 0 \
            else "right" if self.orientation < 0 \
            else "straight"
        return f"Exit({direction},{np.round(self.turn_target, 3)})"

    def get_maneuvers(self) -> List[Maneuver]:
        maneuvers = []
        state = self.start_frame[self.agent_id]
        in_junction = self.scenario_map.junction_at(state.position) is not None
        current_lane = self._find_current_lane(state, in_junction)
        current_distance = current_lane.distance_at(state.position)

        frame = self.start_frame

        connecting_lane = current_lane
        if not in_junction:
            # Follow lane until start of turn if outside of give-way distance
            if current_lane.length - current_distance > ip.GiveWay.GIVE_WAY_DISTANCE + ip.Maneuver.POINT_SPACING:
                distance_of_termination = current_lane.length - ip.GiveWay.GIVE_WAY_DISTANCE
                lane_follow_termination = current_lane.point_at(distance_of_termination)
                config_dict = {
                    "type": "follow-lane",
                    "termination_point": lane_follow_termination
                }
                config = ip.ManeuverConfig(config_dict)
                if self.open_loop:
                    man = ip.FollowLane(config, self.agent_id, frame, self.scenario_map)
                else:
                    man = ip.CLManeuverFactory.create(config, self.agent_id, frame, self.scenario_map)
                maneuvers.append(man)
                frame = ip.Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)

            connecting_lane = self._find_connecting_lane(current_lane)

            # Add give-way maneuver
            config_dict = {
                "type": "give-way",
                "stop": self.stop,
                "termination_point": current_lane.midline.coords[-1],
                "junction_road_id": connecting_lane.parent_road.id,
                "junction_lane_id": connecting_lane.id
            }
            config = ip.ManeuverConfig(config_dict)
            if self.open_loop:
                man = ip.GiveWay(config, self.agent_id, frame, self.scenario_map)
            else:
                man = ip.CLManeuverFactory.create(config, self.agent_id, frame, self.scenario_map)
            maneuvers.append(man)
            frame = ip.Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, man)

        # Add turn
        config_dict = {
            "type": "turn",
            "termination_point": self.turn_target,
            "junction_road_id": connecting_lane.parent_road.id,
            "junction_lane_id": connecting_lane.id
        }
        config = ip.ManeuverConfig(config_dict)
        if self.open_loop:
            man = ip.Turn(config, self.agent_id, frame, self.scenario_map)
        else:
            man = ip.CLManeuverFactory.create(config, self.agent_id, frame, self.scenario_map)
        maneuvers.append(man)
        self.final_frame = ip.Maneuver.play_forward_maneuver(self.agent_id, self.scenario_map, frame, maneuvers[-1])

        return maneuvers

    def _nearest_lane_to_goal(self, lane_list: List[ip.Lane]) -> ip.Lane:
        best_lane = None
        best_distance = np.inf
        for connecting_lane in lane_list:
            distance = np.linalg.norm(self.turn_target - np.array(connecting_lane.midline.coords[-1]))
            if distance < self.TURN_TARGET_THRESHOLD and distance < best_distance:
                best_lane = connecting_lane
                best_distance = distance
        return best_lane

    def _find_current_lane(self, state: ip.AgentState, in_junction: bool) -> ip.Lane:
        if not in_junction:
            return self.scenario_map.best_lane_at(state.position, state.heading)
        all_lanes = self.scenario_map.lanes_at(state.position, max_distance=0.5)
        return self._nearest_lane_to_goal(all_lanes)

    def _find_connecting_lane(self, current_lane: ip.Lane) -> Optional[ip.Lane]:
        return self._nearest_lane_to_goal(current_lane.link.successor)

    @staticmethod
    def applicable(state: ip.AgentState, scenario_map: ip.Map) -> bool:
        """ True if either Turn (in junction) or GiveWay is applicable (ahead of junction) and not on
         a roundabout road. """
        in_junction = scenario_map.junction_at(state.position) is not None
        if in_junction:
            return ip.Turn.applicable(state, scenario_map)
        else:
            # We never need to give way in a roundabout so this should never be applicable.
            #  Instead we Continue until in_junction is True and then execute a single Turn action.
            in_roundabout = scenario_map.in_roundabout(state.position, state.heading)
            return ip.GiveWay.applicable(state, scenario_map) and not in_roundabout

    @staticmethod
    def get_possible_args(state: ip.AgentState, scenario_map: ip.Map, goal: ip.Goal = None) -> List[Dict]:
        """ Return turn endpoint if approaching junction; if in junction
        return all possible turns within angle threshold"""
        targets = []
        junction = scenario_map.junction_at(state.position)

        if junction is not None:
            lane = scenario_map.best_lane_at(state.position, state.heading, max_distance=0.6, goal=goal)
            if lane is None:
                raise ValueError(f"No lane found at {state.position}, {state.heading}, {goal}")
            if junction.in_roundabout:
                potential_junction_lanes = scenario_map.lanes_at(state.position) # this may not all junction lanes
                junction_lanes = []
                for jl in potential_junction_lanes:
                    if jl.parent_road in junction.roads:
                        junction_lanes.append(jl)
                if len(junction_lanes) > 1:
                    lane = [jl for jl in junction_lanes
                            if not scenario_map.road_in_roundabout(jl.parent_road)][0]
            targets.append(np.array(lane.midline.coords[-1]))
        else:
            current_lane = scenario_map.best_lane_at(state.position, state.heading)
            for connecting_lane in current_lane.link.successor:
                if not scenario_map.road_in_roundabout(connecting_lane.parent_road):
                    targets.append(np.array(connecting_lane.midline.coords[-1]))

        return [{"turn_target": t} for t in targets]
