import logging

from igp2.planlibrary.maneuver import Maneuver
from igp2.trajectory import Trajectory
from igp2.planlibrary.maneuver import SwitchLane, GiveWay

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
        return SwitchLane.TARGET_SWITCH_LENGTH

    @target_switch_length.setter
    def target_switch_length(self, value: float):
        isinstance(value, float) and value > 0, f"Minimum switch lane length was {value}."
        SwitchLane.TARGET_SWITCH_LENGTH = value

    @property
    def max_oncoming_vehicle_dist(self):
        return GiveWay.MAX_ONCOMING_VEHICLE_DIST

    @max_oncoming_vehicle_dist.setter
    def max_oncoming_vehicle_dist(self, value: float):
        isinstance(value, float) and value > 0, f"Maximum oncoming vehicle distance for give way was {value}."
        GiveWay.MAX_ONCOMING_VEHICLE_DIST = value
