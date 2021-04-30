import numpy as np
from typing import Union, Tuple, List, Dict


class Frame:
    """ Define the physical state of an agent at a single timestep """
    def __init__(self, time: float, position: Union[Tuple[float, float], np.ndarray],
                 speed: float, acceleration, heading):
        """
        Cteate a Frame object

        Args:
            time: current time in seconds
            position: 2d point specifying current position of vehicle
            speed: current speed in m/s
            acceleration: longitudinal acceleration
            heading: direction the agent is travelling radians
        """
        self.time = time
        self.position = np.array(position)
        self.speed = speed
        self.acceleration = acceleration
        self.heading = heading


class VelocityTrajectory:
    """ Define a trajectory consisting of a 2d path and velocities """

    def __init__(self, path, velocity):
        """ Create a VelocityTrajectory object

        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        self.path = path
        self.velocity = velocity

