import argparse

import carla
import numpy as np
import random
import igp2 as ip


def parse_args():
    config_specification = argparse.ArgumentParser(description="""
    This script runs the full IGP2 system in a map in the CARLA simulator.
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
                                      default="Town01",
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
                                      default=None,
                                      help="Maximum number of iterations to run the simulation for. If not given, "
                                           "then run until the ego reached its goal.",
                                      type=int)
    config_specification.add_argument("--fps",
                                      default=20,
                                      help="framerate of the simulation",
                                      type=int)
    config_specification.add_argument('--plan_period',
                                      default=2.0,
                                      help="Period at which to run the MCTS planner.",
                                      type=float)
    config_specification.add_argument('--launch_process',
                                      default=False,
                                      help="Launch a new process of CARLA instead of using a currently running one.",
                                      action='store_true')

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

    # You can set the starting goal and pose of the ego here
    ego_id = 0
    ego_goal = ip.PointGoal(np.array((137.3, -59.43)), 1.5)
    frame = {
        ego_id: ip.AgentState(time=0,
                              position=np.array([92.21, -100.10]),
                              velocity=np.array([7.0, 0.0]),
                              acceleration=np.array([0.0, 0.0]),
                              heading=np.pi / 2)
    }

    simulation = ip.simcarla.CarlaSim(server=config["server"],
                                      port=config["port"],
                                      map_name=scenario,
                                      xodr=xodr_path,
                                      carla_path=carla_path,
                                      launch_process=config["launch_process"],
                                      rendering=not config["no_rendering"],
                                      record=config["record"],
                                      fps=config["fps"])

    ego_agent = ip.MCTSAgent(agent_id=ego_id,
                             initial_state=frame[ego_id],
                             t_update=config["plan_period"],
                             scenario_map=simulation.scenario_map,
                             goal=ego_goal,
                             fps=config["fps"])
    simulation.add_agent(ego_agent, rolename="ego")

    # Set spectator view point in the server rendering screen.
    location = carla.Location(x=ego_agent.state.position[0], y=-ego_agent.state.position[1], z=50)
    rotation = carla.Rotation(pitch=-70, yaw=-90)
    transform = carla.Transform(location, rotation)
    simulation.spectator.set_transform(transform)

    tm = simulation.get_traffic_manager()
    tm.set_agents_count(config["n_traffic"])
    tm.set_ego_agent(ego_agent)
    # tm.set_spawn_speed(low=4, high=14)
    tm.update(simulation)

    if config["visualiser"]:
        visualiser = ip.simcarla.Visualiser(simulation)
        visualiser.run(config["max_iter"])
    else:
        simulation.run(config["max_iter"])


if __name__ == '__main__':
    main()
