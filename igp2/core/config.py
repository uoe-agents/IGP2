import logging

from igp2.planlibrary.maneuver import Maneuver, Stop
from igp2.core.trajectory import Trajectory
from igp2.planlibrary.maneuver import SwitchLane, GiveWay
from igp2.planlibrary.macro_action import ChangeLane
from igp2.recognition.astar import AStar

logger = logging.getLogger(__name__)


class Configuration:
    """ Class that serves as central location to get or set global IGP2 parameters,
    such as maximum speed or swerving distance. """

    __FPS = 20

    @classmethod
    def set_properties(cls, **kwargs):
        """ Set any properties of IGP2 using a dictionary. """
        for k, v in kwargs.items():
            try:
                getattr(cls, k).fset(cls, v)
            except AttributeError:
                continue

    @property
    def fps(self) -> int:
        """ Framerate of simulation. """
        return self.__FPS

    @fps.setter
    def fps(self, value: int):
        assert isinstance(value, int) and value > 0, f"FPS must be a positive integer and not {value}."
        self.__FPS = value

    @property
    def max_speed(self) -> float:
        """ Global speed limit. """
        return Maneuver.MAX_SPEED

    @max_speed.setter
    def max_speed(self, value: float):
        assert value > 0, f"Maximum speed cannot be {value}."
        Maneuver.MAX_SPEED = value

    @property
    def min_speed(self) -> float:
        """ Minimum speed in turns. """
        return Maneuver.MIN_SPEED

    @min_speed.setter
    def min_speed(self, value: float):
        assert isinstance(value, float) and value > 0, f"Minimum turn speed cannot be {value}."
        Maneuver.MIN_SPEED = value

    @property
    def velocity_stop(self) -> float:
        """ The threshold in m/s to classify velocities as stopped. """
        return Trajectory.VELOCITY_STOP

    @velocity_stop.setter
    def velocity_stop(self, value: float):
        Trajectory.VELOCITY_STOP = value

    @property
    def target_switch_length(self):
        """ The ideal target length for a lane switch. """
        return SwitchLane.TARGET_SWITCH_LENGTH

    @target_switch_length.setter
    def target_switch_length(self, value: float):
        isinstance(value, float) and value > 0, f"Target switch lane length was {value}."
        SwitchLane.TARGET_SWITCH_LENGTH = value

    @property
    def min_switch_length(self):
        """ The minimum target length for a lane switch. """
        return SwitchLane.MIN_SWITCH_LENGTH

    @min_switch_length.setter
    def min_switch_length(self, value: float):
        isinstance(value, float) and value > 0, f"Minimum switch lane length was {value}."
        SwitchLane.MIN_SWITCH_LENGTH = value

    @property
    def max_oncoming_vehicle_dist(self):
        """ The maximum distance for a vehicle to be considered when giving way. """
        return GiveWay.MAX_ONCOMING_VEHICLE_DIST

    @max_oncoming_vehicle_dist.setter
    def max_oncoming_vehicle_dist(self, value: float):
        isinstance(value, float) and value > 0, f"Maximum oncoming vehicle distance for give way was {value}."
        GiveWay.MAX_ONCOMING_VEHICLE_DIST = value

    @property
    def next_lane_offset(self) -> float:
        """ The lane offset used in AStar for getting the next lane. """
        return AStar.NEXT_LANE_OFFSET

    @next_lane_offset.setter
    def next_lane_offset(self, value):
        AStar.NEXT_LANE_OFFSET = value

    @property
    def default_stop_duration(self) -> float:
        """ The default duration for a stop maneuver. """
        return Stop.DEFAULT_STOP_DURATION

    @default_stop_duration.setter
    def default_stop_duration(self, value):
        """ The default duration for a stop maneuver. """
        Stop.DEFAULT_STOP_DURATION = value

    @property
    def give_way_distance(self) -> float:
        """ The distance from a junction at which to begin the GiveWay maneuver. """
        return GiveWay.GIVE_WAY_DISTANCE

    @give_way_distance.setter
    def give_way_distance(self, value):
        """ The distance from the junction at which to begin the GiveWay maneuver."""
        GiveWay.GIVE_WAY_DISTANCE = value

    @property
    def check_oncoming(self) -> bool:
        """ Whether to check for oncoming vehicles when changing lanes."""
        return ChangeLane.CHECK_ONCOMING

    @check_oncoming.setter
    def check_oncoming(self, value: bool):
        ChangeLane.CHECK_ONCOMING = value

    @property
    def check_vehicle_in_front(self) -> bool:
        """ Whether to check for vehicles in front when changing lanes."""
        return Maneuver.CHECK_VEHICLE_IN_FRONT

    @check_vehicle_in_front.setter
    def check_vehicle_in_front(self, value: bool):
        """ Whether to check for vehicles in front when changing lanes."""
        Maneuver.CHECK_VEHICLE_IN_FRONT = value