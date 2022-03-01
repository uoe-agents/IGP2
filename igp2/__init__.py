from .opendrive import *
from .agentstate import AgentState, AgentMetadata
from .trajectory import Trajectory, VelocityTrajectory, StateTrajectory
from .util import Box, Circle
from .goal import Goal, PointGoal, BoxGoal, PointCollectionGoal
from .vehicle import Observation, Action
from .cost import Cost
from .velocitysmoother import VelocitySmoother
from .planlibrary import *
from .recognition import *
from .agents import *
from igp2 import data
from .results import RunResult, MCTSResult, AgentResult, EpisodeResult, \
    PlanningResult, AllMCTSResult, ExperimentResult, RewardResult
from .planning import *
from .vehicle import Vehicle, TrajectoryVehicle, KinematicVehicle
from igp2 import carla


def setup_logging(level=None, vel_smooting_level=None, log_dir=None, log_name=None):
    import sys
    import os
    import logging
    from datetime import datetime

    if level is None:
        level = logging.DEBUG
    if vel_smooting_level is None:
        vel_smooting_level = logging.INFO

    # Add %(asctime)s  for time
    log_formatter = logging.Formatter("[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    root_logger = logging.getLogger("igp2")
    root_logger.setLevel(level)

    if log_dir and log_name:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler("{0}/{1}_{2}.log".format(log_dir, log_name, date_time))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger("igp2.velocitysmoother").setLevel(vel_smooting_level)

    return root_logger
