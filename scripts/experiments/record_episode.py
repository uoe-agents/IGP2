import argparse
import time

import carla
import numpy as np
import random
import igp2 as ip
from igp2.agents.traffic_agent import EgoTrafficAgent


def parse_args():
    config_specification = argparse.ArgumentParser(description="""
    This script records vehicle trajectories in a map in the CARLA simulator.
         """, formatter_class=argparse.RawTextHelpFormatter)

    config_specification.add_argument('--server',
                                      default="localhost",
                                      help="server IP where CARLA is running",
                                      type=str)
    config_specification.add_argument("--port",
                                      default=2000,
                                      help="port where CARLA is accessible on the server.",
                                      type=int)
    config_specification.add_argument('--carla_path', '-p',
                                      default="/opt/carla-simulator",
                                      help="path to the directory where CARLA is installed. "
                                           "Used to launch CARLA if not running.",
                                      type=str)
    config_specification.add_argument('--map', '-m',
                                      default="town01",
                                      help="name of the map (town) to use",
                                      type=str)
    config_specification.add_argument('--seed', '-s',
                                      default=None,
                                      help="random seed to use",
                                      type=int)
    config_specification.add_argument('--no_rendering',
                                      help="whether to disable CARLA rendering",
                                      action="store_true")
    config_specification.add_argument('--record', '-r',
                                      help="whether to create an offline recording of the simulation",
                                      action="store_true")
    config_specification.add_argument('--n_traffic',
                                      default=10,
                                      help="the maximum number of actors to spawn for traffic",
                                      type=int)
    config_specification.add_argument('--visualiser',
                                      default=True,
                                      help="whether to use detailed visualisation for the simulation",
                                      action='store_true')
    config_specification.add_argument('--max_iter',
                                      default=20*60*25,
                                      help="Maximum number of iterations to run the simulation for. If not given, "
                                           "then run until the ego reached its goal.",
                                      type=int)
    config_specification.add_argument("--fps",
                                      default=25,
                                      help="framerate of the simulation",
                                      type=int)
    config_specification.add_argument('--plan_period',
                                      default=2.0,
                                      help="Period at which to run the MCTS planner.",
                                      type=float)

    config_specification.set_defaults(no_rendering=False, record=False)
    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


def main():
    config = parse_args()
    if config["seed"] is not None:
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    ip.setup_logging()
    carla_path = config["carla_path"]
    scenario = config["map"]
    xodr_path = f"scenarios/maps/{scenario.lower()}.xodr"

    simulation = ip.carla.CarlaSim(server=config["server"],
                                   port=config["port"],
                                   map_name=scenario,
                                   xodr=xodr_path,
                                   carla_path=carla_path,
                                   rendering=not config["no_rendering"],
                                   record=config["record"],
                                   fps=config["fps"],
                                   record_trajectories=True)

    tm = simulation.get_traffic_manager()
    tm.set_agents_count(config["n_traffic"])
    tm.spawn_agent(simulation)
    ego_agent = list(tm.agents.values())[0]
    ego_agent.view_radius = 50
    tm.set_ego_agent(ego_agent)
    tm.update(simulation)

    # Set spectator view point in the server rendering screen.
    location = carla.Location(x=ego_agent.state.position[0], y=-ego_agent.state.position[1], z=50)
    rotation = carla.Rotation(pitch=-70, yaw=-90)
    transform = carla.Transform(location, rotation)
    simulation.spectator.set_transform(transform)

    for i in range(config["max_iter"]):
        simulation.step()
        if ego_agent.agent_id not in tm.agents or tm.agents[ego_agent.agent_id] is None:
            break
    simulation.trajectory_history.save_data(0)


if __name__ == '__main__':
    main()
