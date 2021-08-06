import numpy as np
import matplotlib.pyplot as plt
import copy

from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver
from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader
from igp2.trajectory import *
from igp2.velocitysmoother import VelocitySmoother
from igp2.goal import PointGoal
from shapely.geometry import Point
from igp2.recognition.goalrecognition import *
from igp2.recognition.astar import AStar
from igp2.cost import Cost


def update_current_agents(_frame, _current_agents):
    # Iterate over time steps in the episode; Store observed trajectowas tries
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

def remove_offroad_agents(_frame, scenario_map):
    offroad_agent_ids = []
    for key, value in _frame.agents.items():
        position = value.position
        if len(scenario_map.roads_at(position)) == 0:
            offroad_agent_ids.append(key)

    for aid in offroad_agent_ids:
        del _frame.agents[aid]


SCENARIO = "heckstrasse"

if __name__ == '__main__':
    setup_logging()


    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["valid"])
    data_loader.load()

    goals_data = data_loader.scenario.config.goals
    goals = extract_goal_data(goals_data)
    smoother = VelocitySmoother(vmax_m_s=20, n=10, amax_m_s2=5, lambda_acc=10)
    astar = AStar()
    cost = Cost()
    goal_recognition = GoalRecognition(astar=astar, smoother=smoother, cost=cost, scenario_map=scenario_map)

    for episode in data_loader:
        Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit

        # Iterate over each time step and keep track of visible agents' observed trajectories
        current_agents = {}
        output = {}
        for frame in episode.frames:
            update_current_agents(frame, current_agents)
            output[frame.time] = {}
            for aid, agent_state in frame.agents.items():
                if frame.time == 0: last_time = 0
                else: last_time = frame.time - 1
                if aid not in output[last_time]:
                    output[frame.time][aid] = GoalsProbabilities(goals)
                else:
                    output[frame.time][aid] = copy.deepcopy(output[last_time][aid])
                goal_recognition.update_goals_probabilities(output[frame.time][aid], current_agents[aid], aid, frame_ini = episode.frames[current_agents[aid].start_time].agents, frame = frame.agents, maneuver = None)