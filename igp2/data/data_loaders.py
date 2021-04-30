"""
Modified version of code from https://github.com/cbrewitt/av-goal-recognition/blob/master/core/
based on https://github.com/ika-rwth-aachen/drone-dataset-tools
"""

import abc
import logging

from igp2.data.scenario import Scenario, InDScenario

logger = logging.getLogger(__name__)


class DataLoader(abc.ABC):
    """ Abstract class that is implemented by every DataLoader that IGP2 can use. """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._scenario = None

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def load(self, **kwargs) -> Scenario:
        raise NotImplementedError()


class InDDataLoader(DataLoader):
    def load(self) -> InDScenario:
        """ Load all episodes of the scenario  """
        self._scenario = InDScenario.load(self.config_path)
        self._scenario.load_episodes()
        return self._scenario


if __name__ == '__main__':
    from igp2 import setup_logging
    setup_logging()
    loader = InDDataLoader("scenarios/configs/frankenberg.json").load()
