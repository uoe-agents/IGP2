import matplotlib.pyplot as plt
import logging

from igp2 import setup_logging
from igp2.cost import Cost
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

from igp2.agent import AgentState, AgentMetadata
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.macro_action import Exit
from igp2.planning.mcts import MCTS
from igp2.recognition.astar import AStar

# Script to test astar trajectory generation
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.trajectory import VelocityTrajectory, Trajectory
from igp2.velocitysmoother import VelocitySmoother

logger = logging.getLogger(__name__)

SCENARIOS = {
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
    "round": Map.parse_from_opendrive("scenarios/maps/round.xodr"),
}

scenario_map = SCENARIOS["round"]
frame = {
    0: AgentState(time=0,
                  position=np.array([96.8, -0.2]),
                  velocity=4,
                  acceleration=0.0,
                  heading=-2 * np.pi / 3),
    1: AgentState(time=0,
                  position=np.array([25.0, -36.54]),
                  velocity=4,
                  acceleration=0.0,
                  heading=-0.3),
    # 2: AgentState(time=0,
    #               position=np.array([133.75, -61.67]),
    #               velocity=4,
    #               acceleration=0.0,
    #               heading=5 * np.pi / 6),
    # 3: AgentState(time=0,
    #               position=np.array([102.75, -48.31]),
    #               velocity=4,
    #               acceleration=0.0,
    #               heading=np.pi / 2),
}
colors = "rgbyk"

goals = {
    0: PointGoal(np.array([103.99, -5.91]), 2),
    # 1: PointGoal(np.array([103.99, -5.91]), 2),
    # 2: PointGoal(np.array([60.75, -83.77]), 2),
    3: PointGoal(np.array([60.75, -83.77]), 2),
}

cost_factors = {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10, "angular_velocity": 0.0,
                "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}

goal_probabilities = {aid: GoalsProbabilities(goals.values()) for aid in frame.keys()}
astar = AStar(next_lane_offset=0.25)
cost = Cost(factors=cost_factors)
smoother = VelocitySmoother(vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10)
goal_recognition = GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost,
                                   reward_as_difference=True, n_trajectories=2)
mcts = MCTS(scenario_map)

if __name__ == '__main__':
    setup_logging(logging.INFO)

    for agent_id in frame:
        logger.info(f"Running prediction for Agent {agent_id}")
        goal_recognition.update_goals_probabilities(goal_probabilities[agent_id],
                                                    VelocityTrajectory.from_agent_state(frame[agent_id]),
                                                    agent_id, frame, frame, None)

    mcts.search(0, frame, AgentMetadata.default_meta(frame), goal_probabilities)
