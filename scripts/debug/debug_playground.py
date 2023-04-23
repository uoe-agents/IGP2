import os
import random

import logging

import dill
import igp2 as ip
import pickle
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def dump_results(objects, name: str):
    """Saves results binary"""
    filename = name + '.pkl'
    foldername = os.path.dirname(os.path.abspath(__file__)) + '/data/planning_results/'
    filename = foldername + filename

    with open(filename, 'wb') as f:
        dill.dump(objects, f)

SCENARIOS = {
    "heckstrasse": ip.Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
    "round": ip.Map.parse_from_opendrive("scenarios/maps/neuweiler.xodr"),
    "town1": ip.Map.parse_from_opendrive("scenarios/maps/Town01.xodr")
}

round_frame = {
    0: ip.AgentState(time=0,
                     position=np.array([96.8, -0.2]),
                     velocity=4,
                     acceleration=0.0,
                     heading=-2 * np.pi / 3),
    1: ip.AgentState(time=0,
                     position=np.array([25.0, -36.54]),
                     velocity=4,
                     acceleration=0.0,
                     heading=-0.3),
    2: ip.AgentState(time=0,
                     position=np.array([133.75, -61.67]),
                     velocity=4,
                     acceleration=0.0,
                     heading=5 * np.pi / 6),
    3: ip.AgentState(time=0,
                     position=np.array([102.75, -48.31]),
                     velocity=4,
                     acceleration=0.0,
                     heading=np.pi / 2),
}
heckstrasse_frame = frame = {
    0: ip.AgentState(time=0,
                     position=np.array([6.0, 0.7]),
                     velocity=1.5,
                     acceleration=0.0,
                     heading=-0.6),
    1: ip.AgentState(time=0,
                     position=np.array([19.7, -13.5]),
                     velocity=8.5,
                     acceleration=0.0,
                     heading=-0.6),
    2: ip.AgentState(time=0,
                     position=np.array([73.2, -47.1]),
                     velocity=11.5,
                     acceleration=0.0,
                     heading=np.pi - 0.6),
    3: ip.AgentState(time=0,
                     position=np.array([61.35, -13.9]),
                     velocity=5.5,
                     acceleration=0.0,
                     heading=-np.pi + 0.4)
}

round_goals = {
    0: ip.PointGoal(np.array([113.84, -60.6]), 2),
    1: ip.PointGoal(np.array([99.44, -18.1]), 2),
    2: ip.PointGoal(np.array([49.18, -34.4]), 2),
    3: ip.PointGoal(np.array([64.32, -74.3]), 2),
}

heckstrasse_goals = {
    0: ip.PointGoal(np.array([17.40, -4.97]), 2),
    1: ip.PointGoal(np.array([75.18, -56.65]), 2),
    2: ip.PointGoal(np.array([62.47, -17.54]), 2),
}

colors = "rgbyk"

# CHANGE SCENARIOS HERE
scenario_map = SCENARIOS["town1"]
frame = heckstrasse_frame
goals = heckstrasse_goals
ego_id = 0
ego_goal_id = 2

ip.plot_map(scenario_map, markings=True)
for agent_id, state in frame.items():
    plt.plot(*state.position, marker="o")
    plt.text(*state.position, agent_id)
for goal_id, goal in goals.items():
    plt.plot(*goal.center, marker="x")
    plt.text(*goal.center, goal_id)
plt.show()

cost_factors = {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10, "angular_velocity": 0.0,
                "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}

goal_probabilities = {aid: ip.GoalsProbabilities(goals.values()) for aid in frame.keys()}
astar = ip.AStar(next_lane_offset=0.25)
cost = ip.Cost(factors=cost_factors)
smoother = ip.VelocitySmoother(vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10)
goal_recognition = ip.GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost,
                                      reward_as_difference=True, n_trajectories=2)
mcts = ip.MCTS(scenario_map, n_simulations=5, max_depth=7, store_results='final')

if __name__ == '__main__':
    ip.setup_logging()
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    try:
        goal_probabilities = pickle.load(open("preds.p", "rb"))
    except:
        for agent_id in frame:
            logger.info(f"Running prediction for Agent {agent_id}")
            goal_recognition.update_goals_probabilities(goal_probabilities[agent_id],
                                                        ip.VelocityTrajectory.from_agent_state(frame[agent_id]),
                                                        agent_id, frame, frame, None)
        pickle.dump(goal_probabilities, open("preds.p", "wb"))

    mcts.search(ego_id, goals[ego_goal_id], frame, ip.AgentMetadata.default_meta_frame(frame), goal_probabilities)

    experiment_result = ip.PlanningResult(scenario_map, mcts.results, 0.0, frame, goal_probabilities)
    dump_results(experiment_result, 'test_result')

    print("Done")
