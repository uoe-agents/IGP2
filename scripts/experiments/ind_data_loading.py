import numpy as np
import logging
import matplotlib.pyplot as plt

from igp2.core.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.maneuver import Maneuver
from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader
from igp2.recognition.astar import AStar
from igp2.core.trajectory import StateTrajectory


# This script showcases how to load and display data from the dataset

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


SCENARIO = "heckstrasse"
PLOT = True

if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    colors = "brgyk"
    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["valid"])
    data_loader.load()
    goals = np.array(data_loader.scenario.config.goals)
    astar = AStar()
    for episode in data_loader:
        print(f"#agents: {len(episode.agents)}")
        Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit

        # Iterate over each time step and keep track of visible agents' observed trajectories
        current_agents = {}
        for frame in episode.frames:
            if frame.time % 15 != 0:
                continue

            if PLOT: plot_map(scenario_map, markings=True, midline=False)

            for agent_id, state in frame.agents.items():
                print(f"t={frame.time}; aid={agent_id}")
                if PLOT:
                    plt.plot(*state.position, marker="o", color=colors[agent_id % 5])
                    plt.text(*state.position, agent_id)
                for i, goal in enumerate(goals):
                    if PLOT:
                        plt.plot(*goal, marker="x")
                        plt.text(*goal, str(i))
                    trajectories, _ = astar.search(agent_id, frame.agents, PointGoal(goal, 1.0), scenario_map)
                    for traj in trajectories:
                        if PLOT: plt.plot(*list(zip(*traj.path)), color=colors[agent_id % 5])
            if PLOT: plt.show()

        # Iterate over all agents and their full trajectories
        for agent_id, agent in episode.agents.items():
            if agent.goal_reached:
                if PLOT: plot_map(scenario_map, markings=True, midline=True)
                if PLOT: plt.plot(*list(zip(*agent.trajectory.path)), marker="o")
                agent_trajectory = agent.trajectory
                plt.show()
