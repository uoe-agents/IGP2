import pytest

from igp2.planlibrary.maneuver import ManeuverConfig, FollowLane
from igp2.core.goal import PointGoal
from igp2.recognition.goalrecognition import *
from igp2.recognition.astar import AStar
from igp2.core.cost import Cost

scenario_map = Map.parse_from_opendrive(f"scenarios/maps/heckstrasse.xodr")


class TestGoalRecognition:

    def test1(self, trajectory1, goals, goal_types):
        frame = {
            0: AgentState(time=0,
                          position=np.array([41.30, -39.2]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-0.3),
            1: AgentState(time=0,
                          position=np.array([54.21, -50.4]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 5),
            2: AgentState(time=0,
                          position=np.array([64.72, -27.65]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-4 * np.pi / 3),
            3: AgentState(time=0,
                          position=np.array([78.78, -22.10]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=-np.pi / 2 - np.pi / 6),
            4: AgentState(time=0,
                          position=np.array([86.13, -25.47]),
                          velocity=1.5,
                          acceleration=0.0,
                          heading=np.pi / 2),
        }

        trajectory = trajectory1
        goals_probabilities = GoalsProbabilities(goals, goal_types)
        print(goals_probabilities.sample_goals(5))
        smoother = VelocitySmoother()
        astar = AStar()
        cost = Cost()
        goal_recognition = GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost)
        # trajectory.insert(trajectory)
        # print(len(trajectory.heading))
        # print(len(trajectory.velocity))
        print(goals_probabilities)
        goal_recognition.update_goals_probabilities(goals_probabilities, trajectory, 0,
                                                                          frame_ini=frame, frame=frame,
                                                                          maneuver=maneuver_follow_lane)
        print(goals_probabilities)


@pytest.fixture()
def trajectory1():
    velocity = np.array([10., 9.65219087, 9.3043612, 8.95649589, 8.60861904,
                         8.26077406, 7.91295965, 7.56514759, 7.21732575, 6.86950166,
                         6.52166818, 6.17383606, 5.82603222, 5.4782067, 5.13039544,
                         4.78256806, 4.43475039, 4.08692743, 3.73910688, 3.39128599,
                         3.04346456, 2.69564081, 2.3478209, 2., 4.94224523,
                         4.3791341, 5.92469626, 7.14872029, 8.02758482, 8.67158042,
                         9.26201364, 9.58410871])

    path = np.array([[18.2, -9.5],
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
def goals():
    goals_data = [[17.40, -4.97],
                  [75.18, -56.65],
                  [62.47, -17.54]]

    goals = []
    for goal_data in goals_data:
        point = Point(np.array(goal_data))
        goals.append(PointGoal(point, 1.))

    return goals


@pytest.fixture()
def goal_types():
    goal_types = [["n/a"],
                  ["n/a"],
                  ["n/a"]]

    goal_types = None

    return goal_types


@pytest.fixture()
def maneuver_follow_lane():
    config = ManeuverConfig({'type': 'follow-lane',
                             'initial_lane_id': 2,
                             'final_lane_id': 2,
                             'termination_point': (27.1, -19.8)})
    agent_id = 0
    position = np.array((8.4, -6.0))
    heading = -0.6
    speed = 10
    velocity = speed * np.array([np.cos(heading), np.sin(heading)])
    acceleration = np.array([0, 0])
    agent_0_state = AgentState(time=0, position=position, velocity=velocity,
                               acceleration=acceleration, heading=heading)
    frame = {0: agent_0_state}
    maneuver = FollowLane(config, agent_id, frame, scenario_map)
    return maneuver

# @pytest.fixture()
# def frame():


#     return frame
