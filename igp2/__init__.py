import sys
import logging
from datetime import datetime


def setup_logging(level=logging.DEBUG, log_dir=None, log_name=None):
    # Add %(asctime)s  for time
    log_formatter = logging.Formatter("[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    root_logger = logging.getLogger("igp2")
    root_logger.setLevel(level)

    if log_dir and log_name:
        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler("{0}/{1}_{2}.log".format(log_dir, log_name, date_time))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger
