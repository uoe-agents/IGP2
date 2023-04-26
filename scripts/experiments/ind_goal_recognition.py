import logging
import numpy as np
import matplotlib.pyplot as plt

from igp2 import setup_logging
from igp2.core.cost import Cost
from igp2.data.data_loaders import InDDataLoader
from igp2.core.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.maneuver import Maneuver
from igp2.recognition.astar import AStar
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.core.velocitysmoother import VelocitySmoother

# This script showcases how to run goal recognition on the InD Dataset

SCENARIO = "round"
PLOT = True

if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    colors = "brgyk"
    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["valid"])
    data_loader.load()
    goals = np.array(data_loader.scenario.config.goals)
    goals = [PointGoal(center, 1.5) for center in goals]

    cost_factors = {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10,
                    "angular_velocity": 0.0,
                    "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}

    astar = AStar(next_lane_offset=0.25)
    cost = Cost(factors=cost_factors)
    smoother = VelocitySmoother(vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10)
    goal_recognition = GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost,
                                       reward_as_difference=True, n_trajectories=2)

    for episode in data_loader:
        print(f"#agents: {len(episode.agents)}")
        Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit

        # Iterate over all agents and their full trajectories
        for agent_id, agent in episode.agents.items():
            goal_probabilities = GoalsProbabilities(goals)
            start_t = agent.trajectory.states[0].time
            end_t = start_t + len(agent.trajectory.states) // 8
            initial_frame = episode.frames[start_t].agents
            end_frame = episode.frames[end_t].agents

            if agent_id not in initial_frame or agent_id not in end_frame:
                continue

            observed_trajectory = agent.trajectory.slice(0, end_t - start_t)
            goal_recognition.update_goals_probabilities(goal_probabilities,
                                                        observed_trajectory=observed_trajectory,
                                                        agent_id=agent_id,
                                                        frame_ini=initial_frame,
                                                        frame=end_frame,
                                                        maneuver=None)

            if PLOT:
                plot_map(scenario_map, markings=True)
                plt.plot(*end_frame[agent_id].position, marker="x")
                plt.plot(*list(zip(*observed_trajectory.path)), marker="o")
                for goal_type in goal_probabilities.goals_and_types:
                    plt.text(*goal_type[0].center, f"p={goal_probabilities.goals_probabilities[goal_type]:.3f}")
                plt.show()
