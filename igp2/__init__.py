"""
IGP2: Interpretable Goal-Based Prediction and Planning for Autonomous Vehicles
"""
import logging
logger = logging.getLogger(__name__)

from .opendrive import *
from .core import *
from .planlibrary import *
from .recognition import *
from .agents import *
from .planning import *
from .agents.mcts_agent import MCTSAgent
from igp2 import data

try:
    from igp2 import carlasim
except ImportError as e:
    logger.debug("CARLA is not installed.")

from igp2 import simplesim


try:
    import gymnasium as gym
    gym.register(
        id="igp2-v0",
        entry_point=simplesim.SimulationEnv
    )
except ImportError as e:
    logger.debug("Gymnasium is not installed.")


def setup_logging(level=None, vel_smooting_level=None, log_dir=None, log_name=None):
    import sys
    import os
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

    logging.getLogger("igp2.core.velocitysmoother").setLevel(vel_smooting_level)

    return root_logger
