from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader

import matplotlib.pyplot as plt

from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.macro_action import Continue, ChangeLaneLeft
from igp2.planlibrary.maneuver import Maneuver, SwitchLaneLeft

SCENARIO = "heckstrasse"

if __name__ == '__main__':
    setup_logging()

    scenario_map = Map.parse_from_opendrive("scenarios/maps/test.xodr")
    plot_map(scenario_map, markings=True)
    plt.show()

    data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["train"])
    data_loader.load()
    for episode in data_loader:
        Maneuver.MAX_SPEED = episode.metadata.max_speed
        for frame in episode.frames:
            traj = ChangeLaneLeft(0, frame.agents, data_loader.scenario.opendrive_map).get_trajectory().path
            plot_map(data_loader.scenario.opendrive_map)
            plt.plot(traj[:, 0], traj[:, 1])
            plt.show()
