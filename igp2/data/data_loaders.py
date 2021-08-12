"""
Modified version of code from https://github.com/cbrewitt/av-goal-recognition/blob/master/core/
based on https://github.com/ika-rwth-aachen/drone-dataset-tools
"""

import abc
import logging
from typing import Optional, List
from itertools import compress

from igp2.data.episode import Episode
from igp2.data.scenario import Scenario, InDScenario

logger = logging.getLogger(__name__)


class DataLoader(abc.ABC):
    """ Abstract class that is implemented by every DataLoader that IGP2 can use.

    A set of recordings are collected into an Episode. Episodes and the corresponding Map and configuration
    are managed by a Scenario object. The created DataLoader is iterable.
    """
    def __init__(self, config_path: str, splits: List[str] = None):
        """ Create a new data loader object

        Args:
            config_path: The path under which the configuration JSON file is located
            splits: Optional parameter to specify which data split(s) to iterate over.
        """
        self.config_path = config_path
        self.splits = splits
        self._scenario = None

    def __iter__(self):
        raise NotImplementedError

    def __next__(self) -> Episode:
        raise NotImplementedError

    @property
    def scenario(self) -> Optional[Scenario]:
        """ Return the Scenario object"""
        return self._scenario

    def load(self):
        """ Load the Scenario object with the configuration file."""
        raise NotImplementedError

    def train(self) -> List[Episode]:
        """ Return the training data portion of the Scenario """
        raise NotImplementedError

    def valid(self) -> List[Episode]:
        """ Return the validation data portion of the Scenario """
        raise NotImplementedError

    def test(self) -> List[Episode]:
        """ Return the test data portion of the Scenario """
        raise NotImplementedError


class InDDataLoader(DataLoader):
    def load(self):
        """ Load all episodes of the scenario  """
        self._scenario = InDScenario.load(self.config_path, self.splits)

    def __iter__(self):
        if self._scenario is None:
            raise RuntimeError("The scenario has not been loaded yet. Try calling the load() method!")

        self._iter_idx = 0
        return self

    def __next__(self) -> Episode:
        if self._iter_idx < len(self._scenario.episodes):
            episode = self._scenario.episodes[self._iter_idx]
            self._iter_idx += 1
            return episode
        else:
            raise StopIteration

    def get_split(self, splits: List[str] = None) -> List[Episode]:
        if self._scenario is None:
            raise RuntimeError("The scenario has not been loaded yet. Try calling the load() method!")

        if splits is None:
            return self._scenario.episodes
        else:
            indices = []
            for s in splits:
                if s not in self.splits:
                    raise ValueError(f"Split type {s} is not in the valid loaded splits: {self.splits}!")
                indices.extend(self._scenario.config.dataset_split[s])
            return list(compress(self._scenario.episodes, indices))

    def train(self) -> List[Episode]:
        return self.get_split(["train"])

    def valid(self) -> List[Episode]:
        return self.get_split(["valid"])

    def test(self) -> List[Episode]:
        return self.get_split(["test"])
