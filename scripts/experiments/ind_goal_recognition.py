import numpy as np
import matplotlib.pyplot as plt

from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver
from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader
from igp2.trajectory import *
from igp2.goal import PointGoal
from shapely.geometry import Point
from igp2.recognition.goalrecognition import *
from igp2.recognition.astar import AStar
from igp2.cost import Cost


def update_current_agents(_frame, _current_agents):
    # Iterate over time steps in the episode; Store observed trajectories
    dead_agent_ids = [aid for aid in _current_agents if aid not in _frame.agents.keys()]
    for aid in dead_agent_ids:
        del _current_agents[aid]

    for aid, state in _frame.agents.items():
        if aid in _current_agents:
            _current_agents[aid].add_state(state)
        else:
            _current_agents[aid] = StateTrajectory(episode.metadata.frame_rate, _frame.time)

def extract_goal_data(goals_data):
    goals = []
    for goal_data in goals_data:
        point = Point(np.array(goal_data))
        goals.append(PointGoal(point, 1.))

    return goals


SCENARIO = "heckstrasse"

if __name__ == '__main__':
    setup_logging()


    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["test"])
    data_loader.load()

    goals_data = data_loader.scenario.config.goals
    goals = extract_goal_data(goals_data)
    smoother = VelocitySmoother(vmax_m_s=20)
    astar = AStar()
    cost = Cost()
    goal_recognition = GoalRecognition(astar=astar, smoother=smoother, cost=cost, scenario_map=scenario_map)

    for episode in data_loader:
        Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit

        # Iterate over each time step and keep track of visible agents' observed trajectories
        current_agents = {}
        frame_ini = episode.frames[0].agents
        goals_probabilities = GoalsProbabilities(goals)
        #print("prior probabilities:", goals_probabilities.goals_probabilities)
        for frame in episode.frames:
            update_current_agents(frame, current_agents)
            agentId = 4
            goals_probabilities = goal_recognition.update_goals_probabilities(goals_probabilities, current_agents[agentId], agentId, frame_ini = frame_ini, frame = frame.agents, maneuver = None)
            print("updated probabilities:", goals_probabilities.goals_probabilities)