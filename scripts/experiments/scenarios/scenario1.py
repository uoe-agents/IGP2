import logging
from typing import Dict, List, Tuple

from shapely.geometry import Polygon

import igp2 as ip
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_random_frame(ego: int,
                          layout: ip.Map,
                          spawn_vel_ranges: List[Tuple[ip.Box, Tuple[float, float]]]) -> Dict[int, ip.AgentState]:
    """ Generate a new frame with randomised spawns and velocities for each vehicle.

    Args:
        ego: The id of the ego
        layout: The road layout
        spawn_vel_ranges: A list of pairs of spawn ranges and velocity ranges.

    Returns:
        A new randomly generated frame
    """
    ret = {}
    for i, (spawn, vel) in enumerate(spawn_vel_ranges, ego):
        poly = Polygon(spawn.boundary)
        best_lane = None
        max_overlap = 0.0
        for road in layout.roads.values():
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    overlap = lane.boundary.intersection(poly).area
                    if overlap > max_overlap:
                        best_lane = lane
                        max_overlap = overlap

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * np.random.random() + start_d
        spawn_position = np.array(best_lane.point_at(position_d))

        ret[i] = ip.AgentState(time=0,
                               position=spawn_position,
                               velocity=(vel[1] - vel[0]) * np.random.random() + vel[0],
                               acceleration=np.array([0.0, 0.0]),
                               heading=best_lane.get_heading_at(position_d))

    return ret


if __name__ == '__main__':
    ip.setup_logging()
    logging.getLogger("igp2.velocitysmoother").setLevel(logging.INFO)
    np.seterr(divide="ignore")

    # Set run parameters here
    seed = 42
    max_speed = 12.0
    ego_id = 0
    n_simulations = 15
    fps = 20  # Simulator frequency
    T = 2  # MCTS update period

    random.seed(seed)
    np.random.seed(42)
    ip.Maneuver.MAX_SPEED = max_speed  # TODO: add global method to igp2 to set all possible parameters

    # Set randomised spawn parameters here
    ego_spawn_box = ip.Box(np.array([-65.0, -1.8]), 10, 3.5, 0.0)
    ego_vel_range = (5.0, max_speed)
    veh1_spawn_box = ip.Box(np.array([-60.0, 1.7]), 10, 3.5, 0.0)
    veh1_vel_range = (5.0, max_speed)
    veh2_spawn_box = ip.Box(np.array([-18.34, -25.5]), 3.5, 10, 0.0)
    veh2_vel_range = (5.0, max_speed)

    # Vehicle goals
    goals = {
        ego_id: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0)),
        ego_id + 1: ip.BoxGoal(ip.Box(np.array([-22, -25.5]), 3.5, 5, 0.0)),
        ego_id + 2: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0))
    }

    scenario_path = "scenarios/maps/scenario1.xodr"
    scenario_map = ip.Map.parse_from_opendrive(scenario_path)

    frame = generate_random_frame(ego_id,
                                  scenario_map,
                                  [(ego_spawn_box, ego_vel_range),
                                   (veh1_spawn_box, veh1_vel_range),
                                   (veh2_spawn_box, veh2_vel_range)])

    # ip.plot_map(scenario_map, markings=True, midline=True)
    # plt.plot(*list(zip(*ego_spawn_box.boundary)))
    # plt.plot(*list(zip(*veh1_spawn_box.boundary)))
    # plt.plot(*list(zip(*veh2_spawn_box.boundary)))
    # for aid, state in frame.items():
    #     plt.plot(*state.position, marker="x")
    #     plt.text(*state.position, aid)
    # for goal in goals.values():
    #     plt.plot(*list(zip(*goal.box.boundary)), c="g")
    # plt.gca().add_patch(plt.Circle(frame[0].position, 50, color='b', fill=False))
    # plt.show()

    cost_factors = {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10,
                    "angular_velocity": 0.0, "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}

    carla_sim = ip.carla.CarlaSim(xodr='scenarios/maps/scenario1.xodr',
                                  carla_path="C:\\Carla")

    agents = {}
    agents_meta = ip.AgentMetadata.default_meta_frame(frame)
    for aid in frame.keys():
        goal = goals[aid]

        if aid == ego_id:
            agents[aid] = ip.MCTSAgent(agent_id=aid,
                                       initial_state=frame[aid],
                                       t_update=T,
                                       scenario_map=scenario_map,
                                       goal=goal,
                                       cost_factors=cost_factors,
                                       fps=fps,
                                       n_simulations=n_simulations)
        else:
            agents[aid] = ip.carla.TrafficAgent(aid, frame[aid], goal, fps)
            agents[aid].set_destination(goal, scenario_map)

        carla_sim.add_agent(agents[aid])

    carla_sim.attach_camera(carla_sim.agents[ego_id].actor)
    carla_sim.run()
