import logging
import os.path
import sys

import igp2 as ip
import numpy as np
import random
import matplotlib.pyplot as plt

from util import generate_random_frame, parse_args, load_config, to_ma_list, setup_logging
from igp2.core.config import Configuration

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    config = load_config(args)

    setup_logging(debug=args.debug, log_path=args.save_log_path)

    logger.debug(args)

    seed = args.seed if args.seed else config["scenario"]["seed"] if "seed" in config["scenario"] else 21

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip_config = Configuration()
    ip_config.set_properties(**config["scenario"])

    xodr_path = config["scenario"]["map_path"]
    scenario_map = ip.Map.parse_from_opendrive(xodr_path)

    frame = generate_random_frame(scenario_map, config)

    if args.plot_map_only:
        ip.plot_map(scenario_map, hide_road_bounds_in_junction=True, markings=True)
        for aid, state in frame.items():
            plt.plot(*state.position, marker="o")
        plt.show()
        return True

    simulation = None
    try:
        fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20

        if args.carla:
            map_name = os.path.split(xodr_path)[1][:-5]
            if args.map != map_name:
                logger.warning("Map name is not equal to the XODR name. This will likely cause issues.")
            simulation = ip.carlasim.CarlaSim(
                server=args.server, port=args.port,
                map_name=args.map, xodr=scenario_map,
                carla_path=args.carla_path, launch_process=args.launch_process,
                rendering=not args.no_rendering, record=args.record, fps=fps)
        else:
            simulation = ip.simplesim.Simulation(scenario_map, fps)

        ego_agent = None
        for agent_config in config["agents"]:
            agent, rolename = create_agent(agent_config, scenario_map, frame, fps, args)
            simulation.add_agent(agent, rolename=rolename)
            if rolename == "ego":
                ego_agent = agent

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
    if "n_traffic" in config["scenario"] and config["scenario"]["n_traffic"] > 0:
        tm = simulation.get_traffic_manager()
        tm.set_agents_count(config["scenario"]["n_traffic"])
        tm.set_ego_agent(ego_agent)
        # tm.set_spawn_speed(low=4, high=14)
        tm.update(simulation)

    if not args.no_visualiser:
        visualiser = ip.carlasim.Visualiser(simulation, record=args.record_visualiser)
        visualiser.run(config["scenario"]["max_steps"])
    else:
        simulation.run(config["scenario"]["max_steps"])
    return True


def run_simple_simulation(simulation, args, config) -> bool:
    for t in range(config["scenario"]["max_steps"]):
        alive = simulation.step()
        if args.plot is not None and t % args.plot == 0:
            ip.simplesim.plot_simulation(simulation, debug=False)
            plt.show()
        if not alive:
            return False
    return True


def create_agent(agent_config, scenario_map, frame, fps, args):
    base_agent = {"agent_id": agent_config["id"], "initial_state": frame[agent_config["id"]],
                  "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])), "fps": fps}

    mcts_agent = {"scenario_map": scenario_map,
                  "cost_factors": agent_config.get("cost_factors", None),
                  "view_radius": agent_config.get("view_radius", None),
                  "kinematic": not args.carla,
                  "velocity_smoother": agent_config.get("velocity_smoother", None),
                  "goal_recognition": agent_config.get("goal_recognition", None),
                  "stop_goals": agent_config.get("stop_goals", False)}

    if agent_config["type"] == "MCTSAgent":
        agent = ip.MCTSAgent(**base_agent, **mcts_agent, **agent_config["mcts"])
        rolename = "ego"
    elif agent_config["type"] == "TrafficAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        rolename = agent_config.get("rolename", "car")
        agent = ip.TrafficAgent(**base_agent)
    else:
        raise ValueError(f"Unsupported agent type {agent_config['type']}")
    return agent, rolename


if __name__ == '__main__':
    sys.exit(main())
