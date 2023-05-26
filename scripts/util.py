import sys
import os
from typing import Dict, List, Any

import logging
import igp2 as ip
import numpy as np
import argparse
import json
from shapely.geometry import Polygon
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
    This script runs the full IGP2 system in a selected map or scenario using either the CARLA simulator or a
    simpler simulator for rapid testing, but less realistic simulations.
         """, formatter_class=argparse.RawTextHelpFormatter)

    # --------General arguments-------------#
    parser.add_argument('--save_log_path',
                        type=str,
                        default=None,
                        help='save log to the specified path')
    parser.add_argument('--seed',
                        default=None,
                        help="random seed to use",
                        type=int)
    parser.add_argument("--fps",
                        default=20,
                        help="framerate of the simulation",
                        type=int)
    parser.add_argument("--config_path",
                        type=str,
                        help="path to a configuration file.")
    parser.add_argument('--record', '-r',
                        help="whether to create an offline recording of the simulation",
                        action="store_true")  # TODO: Not implemented for simple simulator yet.
    parser.add_argument("--debug",
                        action="store_true",
                        default=False,
                        help="whether to display debugging plots and logging commands")
    parser.add_argument("--plot_map_only",
                        action="store_true",
                        default=False,
                        help="if true, only plot the scenario map and a random then exit the program")
    parser.add_argument("--plot",
                        type=int,
                        default=None,
                        help="display plots of the simulation with this period"
                             " when using the simple simulator.")
    parser.add_argument('--map', '-m',
                        default=None,
                        help="name of the map to use",
                        type=str)

    # -------Simulator specific config---------#
    parser.add_argument("--carla",
                        action="store_true",
                        default=False,
                        help="whether to use CARLA as the simulator instead of the simple simulator.")
    parser.add_argument('--server',
                        default="localhost",
                        help="server IP where CARLA is running",
                        type=str)
    parser.add_argument("--port",
                        default=2000,
                        help="port where CARLA is accessible on the server.",
                        type=int)
    parser.add_argument('--carla_path', '-p',
                        default="/opt/carla-simulator",
                        help="path to the directory where CARLA is installed. "
                             "Used to launch CARLA if not running.",
                        type=str)
    parser.add_argument('--launch_process',
                        default=False,
                        help="use this flag to launch a new process of CARLA instead of using a currently running one.",
                        action='store_true')
    parser.add_argument('--no_rendering',
                        help="whether to disable CARLA rendering",
                        action="store_true")
    parser.add_argument('--no_visualiser',
                        default=False,
                        help="whether to use detailed visualisation for the simulation",
                        action='store_true')
    parser.add_argument('--record_visualiser',
                        default=False,
                        help="whether to use store the PyGame surface during visualisation",
                        action='store_true')

    args = parser.parse_args()
    if args.plot is not None and args.carla:
        logger.debug("--plot is ignored when --carla is used.")
    if args.no_visualiser and args.record_visualiser:
        logger.debug("Using --no_visualiser with --record_visualiser. Latter option will be ignored.")
    return args


def setup_logging(main_logger: logging.Logger = None, debug: bool = False, log_path: str = None):
    # Add %(asctime)s  for time
    level = logging.DEBUG if debug else logging.INFO

    logging.getLogger("igp2.core.velocitysmoother").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    log_formatter = logging.Formatter("[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger("")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if main_logger is not None:
        main_logger.setLevel(level)
        main_logger.addHandler(console_handler)

    if log_path:
        if not os.path.isdir(log_path):
            raise FileNotFoundError(f"Logging path {log_path} does not exist.")

        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f"{log_path}/{date_time}.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)


def load_config(args):
    if "map" in args and args.map is not None:
        path = os.path.join("scenarios", "configs", f"{args.map}.json")
    elif "config_path" in args and args.config_path is not None:
        path = args.config_path
    else:
        raise ValueError("No scenario was specified! Provide either --map or --config_path.")

    try:
        return json.load(open(path, "r"))
    except FileNotFoundError as e:
        logger.exception(msg="No configuration file was found for the given arguments", exc_info=e)
        raise e


def to_ma_list(ma_confs: List[Dict[str, Any]], agent_id: int,
               start_frame: Dict[int, ip.AgentState], scenario_map: ip.Map) \
        -> List[ip.MacroAction]:
    mas = []
    for config in ma_confs:
        config["open_loop"] = False
        frame = start_frame if not mas else mas[-1].final_frame
        if "target_sequence" in config:
            config["target_sequence"] = [scenario_map.get_lane(rid, lid) for rid, lid in config["target_sequence"]]
        ma = ip.MacroActionFactory.create(ip.MacroActionConfig(config), agent_id, frame, scenario_map)
        mas.append(ma)
    return mas


def generate_random_frame(layout: ip.Map, config) -> Dict[int, ip.AgentState]:
    """ Generate a new frame with randomised spawns and velocities for each vehicle.

    Args:
        layout: The current road layout
        config: Dictionary of properties describing agent spawns.

    Returns:
        A new randomly generated frame
    """
    if "agents" not in config:
        return {}

    ret = {}
    for agent in config["agents"]:
        spawn_box = ip.Box(**agent["spawn"]["box"])
        spawn_vel = agent["spawn"]["velocity"]

        poly = Polygon(spawn_box.boundary)
        best_lane = None
        max_overlap = 0.0
        for road in layout.roads.values():
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    overlap = lane.boundary.intersection(poly)
                    if not overlap.is_empty and overlap.area > max_overlap:
                        best_lane = lane
                        max_overlap = overlap.area

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * np.random.random() + start_d

        spawn_position = best_lane.point_at(position_d)
        spawn_heading = best_lane.get_heading_at(position_d)

        vel = (spawn_vel[1] - spawn_vel[0]) * np.random.random() + spawn_vel[0]
        vel = min(vel, ip.Maneuver.MAX_SPEED)
        spawn_velocity = vel * np.array([np.cos(spawn_heading), np.sin(spawn_heading)])

        agent_metadata = ip.AgentMetadata(**agent["metadata"]) if "metadata" in agent \
            else ip.AgentMetadata(**ip.AgentMetadata.CAR_DEFAULT)

        ret[agent["id"]] = ip.AgentState(
            time=0,
            position=spawn_position,
            velocity=spawn_velocity,
            acceleration=np.array([0.0, 0.0]),
            heading=spawn_heading,
            metadata=agent_metadata)
    return ret
