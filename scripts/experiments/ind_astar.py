from shapely.geometry import Point

from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.maneuver import Maneuver
import matplotlib.pyplot as plt
import numpy as np

from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader
from igp2.core.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.recognition.astar import AStar
from igp2.core.trajectory import StateTrajectory


# This experiment showcases the astar trajectory generation.

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
PLOT = False

if __name__ == '__main__':
    setup_logging()

    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["valid"])
    data_loader.load()

    goals_data = data_loader.scenario.config.goals
    goals = extract_goal_data(goals_data)
    astar = AStar()

    colors = "rgbky" * 5

    for episode in data_loader:
        Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit

        # Iterate over each time step and keep track of visible agents' observed trajectories
        for frame in episode.frames:
            PLOT = frame.time > 150
            if PLOT: plot_map(scenario_map, markings=True)
            for agent_id, agent in frame.agents.items():
                c = colors[agent_id]
                if PLOT: plt.plot(*agent.position, color=c, marker="o")
                for goal in goals:
                    trajectories, actions = astar.search(agent_id, frame.agents, goal, scenario_map)
                    if PLOT: plt.plot(*goal.center, marker="x")
                    for trajectory in trajectories:
                        if PLOT: plt.plot(*list(zip(*trajectory.path)), color=c)
            if PLOT: plt.show()
