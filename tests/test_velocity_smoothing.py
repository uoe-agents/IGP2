import pytest
import numpy as np
import logging

from igp2.core.trajectory import VelocityTrajectory
from igp2.core.velocitysmoother import VelocitySmoother
from igp2 import setup_logging


class TestOptimiser:

    def test_optimiser_constraints(self, trajectory1):
        """Checks that all constraints are satisfied"""
        eps = 1e-5
        trajectory1_smooth = VelocitySmoother()
        trajectory1_smooth.load_trajectory(trajectory1)
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

class TestSmoothVelocity:

    def test_endpoint(self, trajectory1):
        """Checks that the smooth_velocity method will only complete after fully bounding the trajectory"""
        trajectory1_smooth = VelocitySmoother()
        trajectory1_smooth.load_trajectory(trajectory1)
        sol, X, V = trajectory1_smooth.smooth_velocity(trajectory1.pathlength, trajectory1.velocity)

        assert X[-1] >= trajectory1.pathlength[-1]

    def test_exception_handling(self):
        logger = setup_logging(level=logging.DEBUG)

        velocity = np.array([np.inf, np.inf, np.inf])
        path = np.array([[0,0], [0,1], [0,2]])
        trajectory = VelocityTrajectory(path, velocity)

        smoother = VelocitySmoother()
        smoother.load_trajectory(trajectory)
        sol, X, V = smoother.smooth_velocity(trajectory.pathlength, trajectory.velocity)

        assert all(trajectory.velocity == V)

class TestSplitSmooth:

    def test_stop_locations(self, trajectory2):
        stop_indent = 27
        stop_length = 2

        trajectory2_smooth = VelocitySmoother(dt_s=0.1, n=10, amax_m_s2=10)
        trajectory2_smooth.load_trajectory(trajectory2)
        smoothed_velocity = trajectory2_smooth.split_smooth(False)

        stop_velocities = smoothed_velocity.tolist()[stop_indent:stop_indent+stop_length]
        expected_stop_velocities = trajectory2.velocity.tolist()[stop_indent:stop_indent+stop_length]

        assert stop_velocities == expected_stop_velocities

class TestSplitAtStops:

    def prep_tests(self, pathlength, velocity, trajectory1):
        trajectory1._pathlength = pathlength
        trajectory1._velocity = velocity
        trajectory = VelocitySmoother()
        trajectory.load_trajectory(trajectory1)

        return trajectory

    @pytest.mark.parametrize(
    "pathlength, split_pathlength, velocity, split_velocity",
    [
        (
        [0., 1., 2., 3., 4., 5., 6.], [[0., 1., 2.], [3.], [4., 5., 6.]],
        [2., 2., 2., 0.0, 2., 2., 2.], [[2., 2., 2.], [0.0], [2., 2., 2.]]
        ),
        (
        [0., 1., 2., 3., 4., 5., 6.], [[0.], [1., 2., 3., 4., 5., 6.]],
        [0., 2., 2., 2., 2., 2., 2.], [[0.0], [2., 2., 2., 2., 2., 2.]]
        ),
        (
        [0., 1., 2., 3., 4., 5., 6.], [[0., 1., 2., 3., 4., 5.], [6.]],
        [2., 2., 2., 2., 2., 2., 0.], [[2., 2., 2., 2., 2., 2.], [0.0]]
        ),
        (
        [0., 1., 2., 3., 4., 5., 6.], [[0.], [1.], [2.], [3.], [4.], [5.], [6.]],
        [0., 2., 0., 2., 0., 2., 0.], [[0.], [2.], [0.], [2.], [0.], [2.], [0.0]]
        ),
        (
        [0., 1., 2., 3., 4., 5., 6.], [[0., 1.], [2., 3.], [4., 5.], [6.]],
        [2., 2., 0., 0., 2., 2., 0.], [[2., 2.], [0., 0.], [2., 2.], [0.0]]
        ),
    ]
    )
    def test_stop_middle(self, pathlength, split_pathlength, velocity, split_velocity, trajectory1):
        pathlength = np.array(pathlength)
        velocity = np.array(velocity)

        trajectory = self.prep_tests(pathlength, velocity, trajectory1)
        trajectory.split_at_stops()

        actual_split_velocity = [i.tolist() for i in trajectory.split_velocity]
        actual_split_pathlength = [i.tolist() for i in trajectory.split_pathlength]
        print(actual_split_velocity)

        assert len(actual_split_velocity) == len(split_velocity)
        for i, j in zip(actual_split_velocity, split_velocity):
            assert i == j

        assert len(actual_split_pathlength) == len(split_pathlength)
        for i, j in zip(actual_split_pathlength, split_pathlength):
            assert i == j
    
class TestRemoveDuplicates:

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

    velocity = np.array([10.        ,  9.70376973,  9.40757245,  9.1113944 ,  8.81522401,
        8.51905169,  8.22286973,  7.92667218,  7.63045469,  7.33421437,
        7.03794969,  6.74166033,  6.44534702,  6.14901145,  5.8526561 ,
        5.55628412,  5.25989921,  4.96350544,  4.66710718,  4.37070893,
        4.07431516,  3.77793025,  3.48155827,  3.18520292,  2.88886734,
        2.59255404,  2.29626468,  0.04433326,  0.04433326,  2.        ,
                    9.62678407, 9.57373018, 9.68069969, 9.78750449,
       9.89409737, 9.99943996, 9.89296186, 9.78632298, 9.67944252,
       9.57236938, 9.4488691 , 9.12283987, 8.38367969, 7.57496026,
       7.23404898, 7.18869327, 7.21070999, 7.35224889, 7.33261758,
       7.20343788, 7.22313614, 7.26143243, 7.25789308, 7.25734072,
       7.25563972, 7.20264738, 7.25932859, 7.36968017, 7.25429353,
       7.2026179 , 7.25720308, 7.2554152 , 7.25551885, 7.48214368,
       8.33153886, 9.37528621, 9.82659925, 9.86607764, 9.88660455,
       9.90713102, 9.92765364, 9.94817108, 9.96868364, 9.98919277,
       9.99029937, 9.96979036, 9.94927804, 9.95169975, 9.96437946])

    path = np.array([[ 93.6       ,  -6.3       ],
       [ 92.60536803,  -6.64978376],
       [ 91.60855966,  -6.99295579],
       [ 90.60970292,  -7.32990499],
       [ 89.60892583,  -7.6610203 ],
       [ 88.60635642,  -7.98669065],
       [ 87.6021227 ,  -8.30730495],
       [ 86.5963527 ,  -8.62325215],
       [ 85.58917445,  -8.93492115],
       [ 84.58071597,  -9.2427009 ],
       [ 83.57110527,  -9.54698031],
       [ 82.56047039,  -9.84814831],
       [ 81.54893935, -10.14659382],
       [ 80.53664017, -10.44270579],
       [ 79.52370087, -10.73687312],
       [ 78.51024948, -11.02948474],
       [ 77.49641402, -11.32092959],
       [ 76.48232251, -11.61159659],
       [ 75.46810298, -11.90187466],
       [ 74.45388345, -12.19215273],
       [ 73.43979194, -12.48281972],
       [ 72.42595648, -12.77426457],
       [ 71.41250508, -13.0668762 ],
       [ 70.39956579, -13.36104353],
       [ 69.3872666 , -13.65715549],
       [ 68.37573556, -13.95560101],
       [ 67.36510068, -14.25676901],
       [ 66.65837319, -14.46976459],
       [ 66.45645106, -14.53062048],
       [ 66.35548999, -14.56104842],
       [ 65.42343358, -15.03579913],
       [ 64.44084903, -15.35362408],
       [ 63.45345672, -15.65707191],
       [ 62.46246699, -15.94973969],
       [ 61.46909019, -16.23522449],
       [ 60.47453666, -16.51712336],
       [ 59.48001677, -16.79903337],
       [ 58.48674087, -17.08455159],
       [ 57.49591929, -17.37727509],
       [ 56.50876239, -17.68080093],
       [ 55.52648053, -17.99872617],
       [ 54.55028405, -18.33464789],
       [ 53.58215208, -18.69437671],
       [ 52.63457343, -19.10563379],
       [ 51.72863597, -19.60109754],
       [ 50.87553621, -20.18310963],
       [ 50.08040614, -20.84219228],
       [ 49.35690577, -21.57888658],
       [ 48.69896827, -22.37540068],
       [ 48.1161831 , -23.2277396 ],
       [ 47.62148645, -24.134285  ],
       [ 47.21406807, -25.08319881],
       [ 46.89790714, -26.06628634],
       [ 46.67585038, -27.0748156 ],
       [ 46.5498836 , -28.09980787],
       [ 46.52116975, -29.132125  ],
       [ 46.59013577, -30.16253923],
       [ 46.76330714, -31.18080353],
       [ 47.01676898, -32.18207906],
       [ 47.3708895 , -33.15219717],
       [ 47.81489739, -34.08459541],
       [ 48.34486847, -34.97094579],
       [ 48.95605795, -35.80334533],
       [ 49.64317682, -36.57419961],
       [ 50.39967318, -37.27692214],
       [ 51.19956725, -37.93047607],
       [ 52.0049295 , -38.57698492],
       [ 52.81371486, -39.21920045],
       [ 53.62546856, -39.85769306],
       [ 54.43973582, -40.49303314],
       [ 55.25606185, -41.12579109],
       [ 56.07399188, -41.75653727],
       [ 56.89307113, -42.3858421 ],
       [ 57.71284482, -43.01427595],
       [ 58.53285816, -43.64240922],
       [ 59.35265639, -44.2708123 ],
       [ 60.17178472, -44.90005557],
       [ 60.98978837, -45.53070943],
       [ 61.80621256, -46.16334426]])

    trajectory = VelocityTrajectory(path, velocity)
    return trajectory
