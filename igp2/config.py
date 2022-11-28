from igp2.planlibrary.maneuver import Maneuver


class Configuration:
    """ Class that serves as central location to get or set global IGP2 parameters,
    such as maximum speed or swerving distance. """

    @property
    def maximum_velocity(self) -> float:
        """ Global speed limit. """
        return Maneuver.MAX_SPEED

    @maximum_velocity.setter
    def maximum_velocity(self, value: float):
        assert isinstance(value, float) and value > 0, f"Maximum speed cannot be {value}."
        Maneuver.MAX_SPEED = value

    @property
    def minimum_speed(self) -> float:
        """ Minimum speed in turns. """
        return Maneuver.MIN_SPEED

    @minimum_speed.setter
    def minimum_speed(self, value: float):
        assert isinstance(value, float) and value > 0, f"Minimum turn speed cannot be {value}."
        Maneuver.MIN_SPEED = value