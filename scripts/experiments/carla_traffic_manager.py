import argparse

import carla
import numpy as np
import random
import igp2 as ip


def parse_args():
    config_specification = argparse.ArgumentParser(description="""
    This script runs the full IGP2 system in a map in the CARLA simulator.
         """, formatter_class=argparse.RawTextHelpFormatter)

    config_specification.add_argument('--carla_path', default="/opt/carla-simulator",
                                      help="path to the directory where CARLA is installed", type=str)
    config_specification.add_argument('--map', default="town01",
                                      help="name of the map (town) to use",
                                      type=str)
    config_specification.add_argument('--seed', default=None,
                                      help="random seed to use",
                                      type=int)
    config_specification.add_argument('--record',
                                      help="whether to create an offline recording of the simulation",
                                      action="store_true")

    config_specification.set_defaults(record=False)
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
    xodr_path = f"scenarios/maps/{scenario}.xodr"

    frame = {
        0: ip.AgentState(time=0,
                         position=np.array([92.21, -100.10]),
                         velocity=np.array([2.0, 0.0]),
                         acceleration=np.array([0.0, 0.0]),
                         heading=np.pi / 2)
    }

    simulation = ip.carla.CarlaSim(xodr=xodr_path, carla_path=carla_path, rendering=True, world=None,
                                   record=config["record"])

    ego_id = 0
    ego_goal = ip.PointGoal(np.array((137.3, -59.43)), 1.5)
    ego_agent = ip.MCTSAgent(agent_id=ego_id, initial_state=frame[ego_id],
                             t_update=1.0, scenario_map=simulation.scenario_map, goal=ego_goal)
    ego_actor = simulation.add_agent(ego_agent)
    location = carla.Location(x=ego_agent.state.position[0], y=-ego_agent.state.position[1], z=50)
    rotation = carla.Rotation(pitch=-70, yaw=-90)
    transform = carla.Transform(location, rotation)
    simulation.spectator.set_transform(transform)

    tm = simulation.get_traffic_manager()
    tm.set_agents_count(5)
    tm.set_ego_agent(ego_agent)
    tm.set_spawn_speed(low=4, high=14)

    # get ego agent id
    simulation.step()
    ego_agent_actor_id = simulation.agents[ego_id].actor_id  # error: agents dict is empty
    print(f'ego actor id: {ego_agent_actor_id}')

    simulation.run(500)


if __name__ == '__main__':
    main()
