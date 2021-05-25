import numpy as np
import matplotlib.pyplot as plt

from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.macro_action import ChangeLaneLeft
from igp2.planlibrary.maneuver import Maneuver
from igp2 import setup_logging
from igp2.agent import AgentState
from igp2.data.data_loaders import InDDataLoader
from igp2.trajectory import StateTrajectory


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

if __name__ == '__main__':
    setup_logging()

    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["test"])
    data_loader.load()
    for episode in data_loader:
        Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit

        # Iterate over each time step and keep track of visible agents' observed trajectories
        current_agents = {}
        for frame in episode.frames:
            update_current_agents(frame, current_agents)
            # Perform other algos here e.g. prediction

        # Iterate over all agents and their full trajectories
        for agent_id, agent in episode.agents.items():
            agent_trajectory = agent.trajectory