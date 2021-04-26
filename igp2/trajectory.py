

class VelocityTrajectory:
    """ Define a trajectory consisting of a 2d path and velocities"""

    def __init__(self, path, velocity):
        """ Create a VelocityTrajectory object

        Args:
            path: nx2 array containing sequence of points
            velocity: array containing velocity at each point
        """
        self.path = path
        self.velocity = velocity
