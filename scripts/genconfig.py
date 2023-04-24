import argparse
import json
import sys
import traceback
import os
import logging

from util import setup_logging


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

    parser.add_argument("--name",
                        type=str,
                        help="name of the scenario to generate",
                        default="new_scenario")
    parser.add_argument("--n_mcts",
                        type=int,
                        help="number of MCTSAgents in the configuration file",
                        default=1)
    parser.add_argument("--n_traffic",
                        type=int,
                        help="number of TrafficAgents in the configuration file",
                        default=1)
    parser.add_argument("--output_path",
                        type=str,
                        help="output directory of the generated config file",
                        default=os.path.join("scenarios", "configs"))

    return parser.parse_args()


def generate_config_template():
    """ Generate a configuration file template using the given commandline options. """
    setup_logging()

    try:
        args = parse_args()
        output = {"scenario": SCENARIO_BASE, "agents": []}
        output["scenario"]["map_path"] = os.path.join("scenarios", "maps", f"{args.name}.xodr")

        if not os.path.exists(output["scenario"]["map_path"]):
            logger.warning(f"Map {output['scenario']['map_path']} does not exist.")

        for n in range(args.n_mcts):
            agent_dict = AGENTS_BASE.copy()
            agent_dict.update(MCTS_AGENT)
            agent_dict["id"] = n
            output["agents"].append(agent_dict)

        for m in range(args.n_mcts, args.n_mcts + args.n_traffic):
            agent_dict = AGENTS_BASE.copy()
            agent_dict.update(TRAFFIC_AGENT)
            agent_dict["id"] = m
            output["agents"].append(agent_dict)

        output_file = os.path.join(args.output_path, f"{args.name}.json")
        json.dump(output, open(output_file, "w"), indent=2)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False
    return True


if __name__ == '__main__':
    sys.exit(generate_config_template())
