from typing import List, Tuple, Dict

import igp2 as ip
import numpy as np
import argparse
from shapely.geometry import Polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed of the simulation.")
    parser.add_argument("--max_speed", type=float, default=10.0, help="Maximum speed limit of the scenario.")
    parser.add_argument("--n_sim", type=int, default=10, help="Number of rollouts in MCTS.")
    parser.add_argument("--fps", type=int, default=20, help="Framerate of the simulation.")
    parser.add_argument("--period", type=float, default=2.0, help="Update frequency of MCTS in seconds.")
    parser.add_argument("--carla_path", type=str, default="C:\\Carla", help="Path to directory containing CARLA.")
    return parser.parse_args()


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

        speed = (vel[1] - vel[0]) * np.random.random() + vel[0]
        heading = best_lane.get_heading_at(position_d)
        ret[i] = ip.AgentState(time=0,
                               position=spawn_position,
                               velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
                               acceleration=np.array([0.0, 0.0]),
                               heading=heading)

    return ret
