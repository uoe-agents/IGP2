from igp2.planlibrary.maneuver import Maneuver
from igp2.trajectory import Trajectory


class Configuration:
    """ Class that serves as central location to get or set global IGP2 parameters,
    such as maximum speed or swerving distance. """

    @classmethod
    def set_properties(cls, **kwargs):
        """ Set any properties of IGP2 using a dictionary. """
        for k, v in kwargs.items():
            getattr(cls, k).fset(cls, v)

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