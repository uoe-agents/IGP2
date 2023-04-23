import argparse
import json
import sys
import traceback
import os

SCENARIO_BASE = {
    "scenario": {
        "map_path": "",
        "max_speed": 10.0,
        "fps": 20,
        "seed": 42,
        "max_steps": 1000,
        "n_traffic": 5
    }
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
    try:
        args = parse_args()
        output = {}

        output_file = os.path.join(args.output_path, f"{args.name}.json")
        json.dump(output, open(output_file, "w"))
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False
    return True


if __name__ == '__main__':
    sys.exit(generate_config_template())
