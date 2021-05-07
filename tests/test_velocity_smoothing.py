import pytest
import numpy as np

from igp2.trajectory import VelocityTrajectory, VelocitySmoother

class TestVelocitySmoother:

    def test_optimiser_constraints(self, trajectory1):
        """Checks that all constraints are satisfied"""
        eps = 1e-5
        trajectory1_smooth = VelocitySmoother(trajectory1)
        sol, X, V = trajectory1_smooth.smooth_velocity(trajectory1.pathlength, trajectory1.velocity)
        vel_inter = trajectory1_smooth.lin_interpolant(trajectory1.pathlength, trajectory1.velocity)

        c1 = X[0] == trajectory1.pathlength[0]
        c2 = V[0] == trajectory1.velocity[0]
        c3 = V.max() - trajectory1_smooth.vmax < eps
        c4 = V.min() >= 0.0 - eps
        c5 = (np.diff(vel_inter(X) - V) >= 0.0 ).all()
        c6 = (abs(np.diff(X) - V[0:-1] * trajectory1_smooth.dt) <= eps).all()
        c7 =  ( np.abs(np.diff(V)) - trajectory1_smooth.dt * trajectory1_smooth.amax <= eps ).all()

        errors = []
        # replace assertions by conditions
        if not c1:
            errors.append("Initial pathlength is not matching.")
        if not c2:
            errors.append("Initial velocity is not matching.")
        if not c3:
            errors.append("Maximum velocity exceeded.")
        if not c4:
            errors.append("Negative velocity.")
        if not c5:
            errors.append("Velocity higher than unsmoothed interpolation.")
        if not c6:
            errors.append("Inconsistency between path travelled and velocity.")
        if not c7:
            errors.append("Maximum acceleration exceeded.")

        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    def test_split_smooth_stop_locations(self):
        pass

    def test_smooth_velocity_endpoint(self, trajectory1):
        """Checks that the smooth_velocity method will only complete after fully bounding the trajectory"""
        trajectory1_smooth = VelocitySmoother(trajectory1)
        sol, X, V = trajectory1_smooth.smooth_velocity(trajectory1.pathlength, trajectory1.velocity)

        assert X[-1] >= trajectory1.pathlength[-1]

    def test_split_at_stops_stop_locations(self):
        pass

    def test_remove_duplicates(self):
        pass


@pytest.fixture()
def trajectory1():
    velocity = np.array([10.,  9.65219087,  9.3043612,  8.95649589,  8.60861904,
                     8.26077406,  7.91295965,  7.56514759,  7.21732575,  6.86950166,
                     6.52166818,  6.17383606,  5.82603222,  5.4782067,  5.13039544,
                     4.78256806,  4.43475039,  4.08692743,  3.73910688,  3.39128599,
                     3.04346456,  2.69564081,  2.3478209,  2., 4.94224523,
                     4.3791341,  5.92469626, 7.14872029, 8.02758482, 8.67158042,
                     9.26201364,  9.58410871])

    path = np.array([[18.2,  -9.5],
                 [19.08138189, -10.11367683],
                 [19.95412248, -10.73969068],
                 [20.82038646, -11.3748227],
                 [21.68233851, -12.015854],
                 [22.54207464, -12.65969006],
                 [23.40107722, -13.30434721],
                 [24.26047047, -13.94847137],
                 [25.11980136, -14.59272908],
                 [25.97644681, -15.24056476],
                 [26.82814053, -15.89494435],
                 [27.67864037, -16.55086796],
                 [28.53560005, -17.19818417],
                 [29.40105998, -17.8342039],
                 [30.26174808, -18.47659271],
                 [31.11446112, -19.12961246],
                 [31.97113869, -19.7773728],
                 [32.83288111, -20.41840715],
                 [33.69324285, -21.06128093],
                 [34.55453678, -21.702907],
                 [35.41472575, -22.34601647],
                 [36.2738406, -22.99057206],
                 [37.13002855, -23.63899089],
                 [37.98267959, -24.2920587],
                 [38.84510881, -24.87269119],
                 [39.8773382, -25.09510637],
                 [40.97168375, -25.10699908],
                 [42.10314107, -24.94812288],
                 [43.24670576, -24.65823132],
                 [44.37737341, -24.27707795],
                 [45.47013963, -23.84441633],
                 [46.5, -23.4]
                 ])

    trajectory = VelocityTrajectory(path, velocity)
    return trajectory
    
@pytest.fixture()
def trajectory2():
    """Trajectory with stop"""

    velocity = np.array([1., 1., 1., 1., 1., 0.0, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    pathlength = np.array([0., 1., 2., 3., 4., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])

    trajectory = VelocityTrajectory(path, velocity)
    return trajectory
