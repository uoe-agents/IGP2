import json
import abc
import logging
from typing import List, Tuple, Dict

from igp2.data.episode import EpisodeConfig, EpisodeLoader, Episode
from igp2.opendrive.map import Map

logger = logging.getLogger(__name__)


class ScenarioConfig:
    """Metadata about a scenario used for goal recognition"""

    def __init__(self, config_dict):
        self.config_dict = config_dict

    @classmethod
    def load(cls, file_path):
        """Loads the scenario metadata into from a json file

        Args:
            file_path (str): path to the file to load

        Returns:
            ScenarioConfig: metadata about the scenario

        """
        with open(file_path) as f:
            scenario_meta_dict = json.load(f)
        return cls(scenario_meta_dict)

    @property
    def goals(self) -> List[Tuple[int, int]]:
        """Possible goals for agents in this scenario"""
        return self.config_dict.get('goals')

    @property
    def name(self) -> str:
        """Name of the scenario"""
        return self.config_dict.get('name')

    @property
    def goal_types(self) -> List[List[str]]:
        """ Possible goals for agents in this scenario"""
        return self.config_dict.get('goal_types')

    @property
    def opendrive_file(self) -> str:
        """ Path to the *.xodr file specifying the OpenDrive map"""
        return self.config_dict.get('opendrive_file')

    @property
    def lat_origin(self) -> float:
        """Latitude of the origin"""
        return self.config_dict.get('lat_origin')

    @property
    def lon_origin(self) -> float:
        """ Longitude of the origin"""
        return self.config_dict.get('lon_origin')

    @property
    def data_format(self) -> str:
        """Format in which the data is stored"""
        return self.config_dict.get('data_format')

    @property
    def data_root(self) -> str:
        """ Path to directory in which the data is stored"""
        return self.config_dict.get('data_root')

    @property
    def episodes(self) -> List[EpisodeConfig]:
        """list of dict: Configuration for all episodes for this scenario"""
        return [EpisodeConfig(c) for c in self.config_dict.get('episodes')]

    @property
    def background_image(self) -> str:
        """Path to background image"""
        return self.config_dict.get('background_image')

    @property
    def background_px_to_meter(self) -> float:
        """ Pixels per meter in background image"""
        return self.config_dict.get('background_px_to_meter')

    @property
    def check_lanes(self) -> bool:
        """ True if Lane data should be checked when loading frames for agents"""
        return self.config_dict.get("check_lanes", False)

    @property
    def reachable_pairs(self) -> List[List[List[float]]]:
        """ Pairs of points, where the second point should be reachable from the first
           Can be used for validating maps"""
        return self.config_dict.get('reachable_pairs')

    @property
    def dataset_split(self) -> Dict[str, List[int]]:
        """ Get the which data split each episode belongs to """
        return self.config_dict.get('dataset_split', None)


class Scenario(abc.ABC):
    """ Represents an arbitrary driving scenario with interactions broken to episodes. """

    def __init__(self, config: ScenarioConfig):
        """ Initialize new Scenario based on the given ScenarioConfig and read map data from config. """
        self.config = config
        self._episodes = None
        self._opendrive_map = None
        self._loader = EpisodeLoader.get_loader(self.config)
        self.load_map()

    def load_map(self):
        if self.config.opendrive_file:
            self._opendrive_map = Map.parse_from_opendrive(self.config.opendrive_file)
        else:
            raise ValueError(f"OpenDrive map was not specified!")

    @property
    def opendrive_map(self) -> Map:
        """ Return the OpenDrive Map of the Scenario. """
        return self._opendrive_map

    @property
    def episodes(self) -> List[Episode]:
        """ Retrieve a list of loaded Episodes. """
        return self._episodes

    @property
    def loader(self) -> EpisodeLoader:
        """ The EpisodeLoader of the Scenario. """
        return self._loader

    @classmethod
    def load(cls, file_path: str, split: List[str] = None):
        """ Initialise a new Scenario from the given config file.

        Args:
            file_path: Path to the file defining the scenario
            split: The data set splits to load as given by indices. If None, load all.

        Returns:
            A new Scenario instance
        """
        raise NotImplementedError


class InDScenario(Scenario):
    @classmethod
    def load(cls, file_path: str, split: List[str] = None):
        config = ScenarioConfig.load(file_path)
        scenario = cls(config)
        scenario.load_episodes(split)
        return scenario

    def load_episodes(self, split: List[str] = None) -> List[Episode]:
        """ Load all/the specified Episodes as given in the ScenarioConfig. Store episodes in field episode """
        if split is not None:
            indices = []
            for s in split:
                indices.extend(self.config.dataset_split[s])
            to_load = [conf for i, conf in enumerate(sorted(self.config.episodes, key=lambda x: x.recording_id))
                       if i in indices]
        else:
            to_load = sorted(self.config.episodes, key=lambda x: x.recording_id)

        logger.info(f"Loading {len(to_load)} episode(s).")
        episodes = []
        for idx, c in enumerate(to_load):
            logger.info(f"Loading Episode {idx + 1}/{len(to_load)}")
            episode = self._loader.load(c, self._opendrive_map if self.config.check_lanes else None)
            episodes.append(episode)

        self._episodes = episodes
        return episodes

    def load_episode(self, episode_id) -> Episode:
        """ Load specific Episode with the given ID. Does not append episode to member field episode. """
        return self._loader.load(EpisodeConfig(self.config.episodes[episode_id]))
