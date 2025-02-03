import argparse
import json
import sys
import os
import logging
import igp2 as ip
import numpy as np
import matplotlib.pyplot as plt

from util import setup_logging
from copy import deepcopy


logger = logging.getLogger(__name__)

SCENARIO_BASE = {
    "map_path": "",
    "max_speed": 15.0,
    "fps": 20,
    "seed": 42,
    "max_steps": 1000,
    "n_traffic": 5
}

AGENTS_BASE = {
    "id": 0,
    "type": "Agent",
    "spawn": {
        "box": {
            "center": [0.0, 0.0],
            "length": 3.5,
            "width": 3.5,
            "heading": 0.0
        },
        "velocity": [
            5.0,
            10.0
        ]
    },
    "goal": {
        "box": {
            "center": [0.0, 0.0],
            "length": 3.5,
            "width": 3.5,
            "heading": 0.0
        }
    }
}

MCTS_AGENT = {
    "type": "MCTSAgent",
    "velocity_smoother": {},
    "goal_recognition": {},
    "cost_factors": {
        "time": 1.0,
        "velocity": 0.0,
        "acceleration": 0.0,
        "jerk": 1.0,
        "heading": 1.0,
        "angular_velocity": 1.0,
        "angular_acceleration": 0.0,
        "curvature": 0.0,
        "safety": 0.0
    },
    "mcts": {
        "t_update": 2.0,
        "n_simulations": 30,
        "max_depth": 5,
        "store_results": "all",
        "trajectory_agents": False,
        "reward_factors": {
            "time": 1.0,
            "jerk": -0.1,
            "angular_velocity": -0.1,
            "curvature": -0.1
        }
    },
    "view_radius": 100,
    "stop_goals": False
}

TRAFFIC_AGENT = {
    "type": "TrafficAgent",
    "macro_actions": []
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
    This script generates an unfilled configuration file template which 
    can be used to define agents within a new scenario.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--name", "-n",
                        type=str,
                        help="name of the scenario to generate",
                        default="new_scenario")
    parser.add_argument("--n_mcts", "-nm",
                        type=int,
                        help="number of MCTSAgents in the configuration file",
                        default=1)
    parser.add_argument("--n_traffic", "-nt",
                        type=int,
                        help="number of TrafficAgents in the configuration file",
                        default=1)
    parser.add_argument("--output_path", "-o",
                        type=str,
                        help="output directory of the generated config file",
                        default=os.path.join("scenarios", "configs"))
    parser.add_argument("--set_locations",
                        action="store_true",
                        help="if true, then load the map and prompt user for spawn and goal locations",
                        default=False)

    return parser.parse_args()


def get_input(aid: int, scenario_map: ip.Map, type_str: str):
    if scenario_map is None:
        return None
    assert type_str in ["spawn", "goal"], f"Input type can only be 'spawn' or 'goal'."

    logger.info(f"Setting {type_str} for agent {aid}. Please follow instructions on the console.")

    fig, ax = plt.subplots()
    ip.plot_map(scenario_map, midline=True, hide_road_bounds_in_junction=True, ax=ax)
    fig.canvas.manager.set_window_title(scenario_map.name)

    # Get spawn
    logger.info("Select on the map the center location (Point 1)"
                " and optionally (close plotting window if not specifying) the heading direction (Point 2)"
                f" for agent {aid}.")

    coords = []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        logger.info(f'Selected: x = {ix}, y = {iy}')

        plt.plot(ix, iy, marker="o")
        plt.show()
        coords.append((ix, iy))

        if len(coords) == 2:
            logger.info("Selected both location and heading point. "
                        "Please close the figure window to continue.")
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    loc = coords[0]
    if len(coords) == 2:
        diff = np.diff(coords, axis=0)[0]
        heading = np.arctan2(diff[1], diff[0])
    else:
        lane = scenario_map.lanes_at(loc)[0]
        ds = lane.distance_at(loc)
        heading = lane.get_heading_at(ds)
    length = float(input("Enter the length of the box in meters(default: 3.5): ") or 3.5)
    width = float(input("Enter the width of the box in meters (default: 3.5): ") or 3.5)
    data = {
        "box": {
            "center": loc,
            "length": length,
            "width": width,
            "heading": heading
        }
    }

    if type_str != "goal":
        min_vel = float(input("Enter the minimum spawn velocity in m/s (default: 5): ") or 5)
        max_vel = float(input("Enter the maximum spawn velocity in m/s (default: 10): ") or 10)
        data["velocity"] = [min_vel, max_vel]

    logger.debug(data)
    return data


def create_dict(scenario_map: ip.Map, set_locations: bool, base_dict: dict, n: int) -> dict:
    agent_dict = deepcopy(AGENTS_BASE)
    agent_dict.update(deepcopy(base_dict))
    agent_dict["id"] = n
    if set_locations:
        spawn = get_input(n, scenario_map, "spawn")
        agent_dict["spawn"].update(spawn)
        goal = get_input(n, scenario_map, "goal")
        agent_dict["goal"].update(goal)
    return agent_dict


def generate_config_template():
    """ Generate a configuration file template using the given commandline options. """
    setup_logging()

    try:
        args = parse_args()

        logger.info(f"Generating config file with name {args.name}")

        output = {"scenario": SCENARIO_BASE, "agents": []}
        output["scenario"]["map_path"] = os.path.join("scenarios", "maps", f"{args.name}.xodr")

        scenario_map = None
        if not os.path.exists(output["scenario"]["map_path"]):
            logger.warning(f"Map {output['scenario']['map_path']} does not exist.")
        elif args.set_locations:
            scenario_map = ip.Map.parse_from_opendrive(output['scenario']['map_path'])

        for n in range(args.n_mcts):
            logger.info(f"Creating MCTSAgent {n}.")
            agent_dict = create_dict(scenario_map, args.set_locations, MCTS_AGENT, n)
            output["agents"].append(agent_dict)

        for m in range(args.n_mcts, args.n_mcts + args.n_traffic):
            logger.info(f"Creating TrafficAgent {m}.")
            agent_dict = create_dict(scenario_map, args.set_locations, TRAFFIC_AGENT, m)
            output["agents"].append(agent_dict)

        output_file = os.path.join(args.output_path, f"{args.name}.json")
        json.dump(output, open(output_file, "w"), indent=2)
    except Exception as e:
        logger.exception(str(e), exc_info=e)
        return False

    logger.info("Done")
    return True


if __name__ == '__main__':
    sys.exit(generate_config_template())
