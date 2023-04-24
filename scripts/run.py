import logging
import os.path
import sys

import igp2 as ip
import numpy as np
import random
import matplotlib.pyplot as plt

from util import generate_random_frame, parse_args, load_config, to_ma_list
from igp2.config import Configuration

logger = logging.Logger(__name__)


def main():
    args = parse_args()
    ip.setup_logging(log_dir=args.save_log_path)
    logger.debug(args)
    config = load_config(args)

    seed = args.seed if args.seed else config["scenario"]["seed"] if "seed" in config["scenario"] else 21

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip_config = Configuration()
    ip_config.set_properties(**config["scenario"])

    xodr_path = config["scenario"]["map_path"]
    scenario_map = ip.Map.parse_from_opendrive(xodr_path)

    frame = {}
    if "agents" in config:
        frame = generate_random_frame(scenario_map, config)

    fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20
    simulation = None
    try:
        if args.carla:
            map_name = os.path.split(xodr_path)[1][:-5]
            if args.map != map_name:
                logger.warning("Map name is not equal to the XODR name. This will likely cause issues.")
            simulation = ip.carla.CarlaSim(
                server=args.server, port=args.port,
                map_name=args.map, xodr=scenario_map,
                carla_path=args.carla_path, launch_process=args.launch_process,
                rendering=not args.no_rendering, record=args.record, fps=fps)
        else:
            simulation = ip.simulator.Simulation(scenario_map, fps)

        ego_agent = None
        for agent in config["agents"]:
            base_agent = {"agent_id": agent["id"], "initial_state": frame[agent["id"]],
                          "goal": ip.BoxGoal(ip.Box(**agent["goal"]["box"])), "fps": fps}

            mcts_agent = {"scenario_map": scenario_map,
                          "cost_factors": agent.get("cost_factors", None),
                          "view_radius": agent.get("view_radius", None),
                          "kinematic": not args.carla,
                          "velocity_smoother": agent.get("velocity_smoother", None),
                          "goal_recognition": agent.get("goal_recognition", None),
                          "stop_goals": agent.get("stop_goals", False)}

            if agent["type"] == "MCTSAgent":
                agent = ip.MCTSAgent(**base_agent, **mcts_agent, **agent["mcts"])
                ego_agent = agent
                rolename = "ego"
            elif agent["type"] == "TrafficAgent":
                if "macro_actions" in agent and agent["macro_actions"]:
                    base_agent["macro_actions"] = to_ma_list(
                        agent["macro_actions"], agent["id"], frame, scenario_map)
                rolename = agent.get("rolename", "car")
                agent = ip.TrafficAgent(**base_agent)
            else:
                raise ValueError(f"Unsupported agent type {agent['type']}")

            simulation.add_agent(agent, rolename=rolename)

        if args.carla:
            result = run_carla_simulation(simulation, ego_agent, args, config)
        else:
            result = run_simple_simulation(simulation, args, config)

    except Exception as e:
        logger.exception(msg=str(e), exc_info=e)
        result = False
    finally:
        if simulation is not None:
            del simulation

    return result


def run_carla_simulation(simulation, ego_agent, args, config) -> bool:
    # Set spectator view point in the server rendering screen.
    tm = simulation.get_traffic_manager()
    tm.set_agents_count(config["scenario"]["n_traffic"])
    tm.set_ego_agent(ego_agent)
    # tm.set_spawn_speed(low=4, high=14)
    tm.update(simulation)

    if not args.no_visualiser:
        visualiser = ip.carla.Visualiser(simulation)
        visualiser.run(config["scenario"]["max_steps"])
    else:
        simulation.run(config["scenario"]["max_steps"])
    return True


def run_simple_simulation(simulation, args, config) -> bool:
    for t in range(config["scenario"]["max_steps"]):
        simulation.step()
        if args.plot is not None and t % args.plot == 0:
            ip.simulator.plot_simulation(simulation, debug=False)
            plt.show()
    return True


if __name__ == '__main__':
    sys.exit(main())
