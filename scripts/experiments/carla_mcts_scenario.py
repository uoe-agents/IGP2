import os
import logging
import dill
import numpy as np
import random
import matplotlib.pyplot as plt

from igp2 import setup_logging
from igp2.agents.agentstate import AgentState, AgentMetadata
from igp2.agents.mcts_agent import MCTSAgent
from igp2.carla.carla_client import CarlaSim
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.recognition.astar import AStar
from igp2.velocitysmoother import VelocitySmoother
from igp2.cost import Cost
from igp2.opendrive.plot_map import plot_map
from igp2.trajectory import VelocityTrajectory
from igp2.results import PlanningResult
from igp2.agents.trajectory_agent import TrajectoryAgent


def dump_results(objects, name: str):
    """Saves results binary"""
    filename = name + '.pkl'
    foldername = os.path.dirname(os.path.abspath(__file__)) + '/data/planning_results/'
    filename = foldername + filename

    with open(filename, 'wb') as f:
        dill.dump(objects, f)


SCENARIOS = {
    "heckstrasse": Map.parse_from_opendrive("scenarios/maps/heckstrasse.xodr"),
}

heading = {0: -0.6,
           1: -0.6,
           2: np.pi,
           3: np.pi + 0.6,
           }

speed = {0: 1.5,
         1: 8.5,
         2: 11.5,
         3: 5.5,
         }

position = {0: np.array([6.0, 0.7]),
            1: np.array([19.7, -13.5]),
            2: np.array([73.2, -47.1]),
            3: np.array([61.7, -15.2]),
            }
heckstrasse_frame ={}
for aid in position.keys():
    heckstrasse_frame[aid] = AgentState(time=0,
                                        position=position[aid],
                                        velocity=speed[aid] * np.array([np.cos(heading[aid]), np.sin(heading[aid])]),
                                        acceleration=np.array([0., 0.]),
                                        heading=heading[aid]
                                        )

heckstrasse_goals = {
    0: PointGoal(np.array([17.40, -4.97]), 2),
    1: PointGoal(np.array([75.18, -56.65]), 2),
    2: PointGoal(np.array([62.47, -17.54]), 2),
}

# CHANGE SCENARIOS HERE
scenario_map = SCENARIOS["heckstrasse"]
frame = heckstrasse_frame
goals = heckstrasse_goals
ego_id = 2
fps = 20
T = 30.0
carla_sim = CarlaSim(xodr='scenarios/maps/heckstrasse.xodr')

# TODO: think of cleaner way
goals_agents = {
    0: 2,
    1: 1,
    2: 0,  # ego goal
    3: 0,
}

# plot_map(scenario_map, markings=True)
# for agent_id, state in frame.items():
#     plt.plot(*state.position, marker="o")
#     plt.text(*state.position, agent_id)
# for goal_id, goal in goals.items():
#     plt.plot(*goal.center, marker="x")
#     plt.text(*goal.center, goal_id)
# plt.show()

# TODO Update
cost_factors = {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10, "angular_velocity": 0.0,
                "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}

# Goal recognition setup:
goal_probabilities = {aid: GoalsProbabilities(goals.values()) for aid in frame.keys()}
astar = AStar(next_lane_offset=0.25)
cost = Cost(factors=cost_factors)
smoother = VelocitySmoother(vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10)
goal_recognition = GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost,
                                   reward_as_difference=True, n_trajectories=2)

if __name__ == '__main__':
    setup_logging()
    seed = 3
    np.random.seed(seed)
    random.seed(seed)

    # structure:
    # - Initialise TrajectoryAgents for Carla: OK
    # - Generate deterministic trajectories for non-ego agents: good way to do it is to run Astar to the goal: OK
    # - initialises the MCTS agent: OK
    # - Repeat, until goal reached or max_step exceeded: OK, in MCTSAgent
    #   - Run MCTS to get a sequence of actions for MCTSAgent
    #   - Run carla for n steps, updating actions @ each timestep for trajectory agent and MCTSAgent
    # - Initialise CarlaServer etc: OK
    agents = {}
    agents_meta = AgentMetadata.default_meta_frame(frame)
    for aid in frame.keys():
        goal = goals[goals_agents[aid]]

        if aid == ego_id:
            agents[aid] = MCTSAgent(agent_id=aid,
                                    initial_state=frame[aid],
                                    t_update=T,
                                    scenario_map=scenario_map,
                                    goal=goal,
                                    cost_factors=cost_factors,
                                    fps=fps,
                                    goals=goals)
        else:
            agents[aid] = TrajectoryAgent(aid, frame[aid], goal, fps)
            trajectories, _ = astar.search(aid, frame, goal, scenario_map, n_trajectories=1)
            trajectory = trajectories[0]
            trajectory.velocity[0] = frame[aid].speed
            smoother.load_trajectory(trajectory)
            trajectory.velocity = smoother.split_smooth()
            agents[aid].set_trajectory(trajectory)

        carla_sim.add_agent(agents[aid])

    carla_sim.run()

    print("Done")
