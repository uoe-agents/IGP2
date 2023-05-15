import abc
import glob
import logging
import os
import numpy as np
import pandas
from typing import List, Dict, Set

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.trajectory import StateTrajectory
from igp2.agents.trajectory_agent import TrajectoryAgent
from igp2.opendrive.map import Map
from igp2.core.util import calculate_multiple_bboxes

logger = logging.getLogger(__name__)


class EpisodeConfig:
    """ Metadata about an episode """

    def __init__(self, config):
        self.config = config

    @property
    def recording_id(self) -> str:
        """ Unique ID identifying the episode"""
        return self.config.get('recording_id')


class EpisodeMetadata:
    def __init__(self, config):
        self.config = config

    @property
    def max_speed(self) -> float:
        """ The speed limit at the episode location. """
        return self.config.get("speedLimit")

    @property
    def frame_rate(self) -> int:
        """ Frame rate of the episode recording. """
        return int(self.config.get("frameRate"))


class EpisodeLoader(abc.ABC):
    """ Abstract class that every EpisodeLoader should represent. Also keeps track of registered subclasses. """
    EPISODE_LOADERS = {}  # Each EpisodeLoader can register its own class as loader here

    def __init__(self, scenario_config):
        self.scenario_config = scenario_config

    def load(self, config: EpisodeConfig, road_map=None, **kwargs):
        raise NotImplementedError()

    @classmethod
    def register_loader(cls, loader_name: str, loader):
        if not issubclass(loader, cls):
            raise ValueError(f"Given loader {loader} is not an EpisodeLoader!")
        if loader_name not in cls.EPISODE_LOADERS:
            cls.EPISODE_LOADERS[loader_name] = loader
        else:
            logger.warning(f"Loader {loader} with name {loader_name} already registered!")

    @classmethod
    def get_loader(cls, scenario_config: "ScenarioConfig") -> "EpisodeLoader":
        """ Get the episode loader as specified within the ScenarioConfig

        Args:
            scenario_config: The scenario configuration

        Returns:
            The corresponding EpisodeLoader
        """
        loader = cls.EPISODE_LOADERS[scenario_config.data_format]
        if loader is None:
            raise ValueError('Invalid data format')
        return loader(scenario_config)


class Frame:
    """ A snapshot of time in the data set"""

    def __init__(self, time: float, dead_ids: Set[int] = None):
        """ Create a new frame.

        Args:
            time: Time of the frame recording
            dead_ids: These agents are treated as dead
        """
        self.time = time
        self.dead_ids = dead_ids if dead_ids is not None else set()
        self._agents = {}

    @property
    def all_agents(self) -> Dict[int, AgentState]:
        return self._agents

    @property
    def agents(self) -> Dict[int, AgentState]:
        return {k: v for k, v in self._agents.items() if k not in self.dead_ids}

    def add_agent_state(self, agent_id: int, state: AgentState):
        """ Add a new agent with its specified state.

        Args:
            agent_id: The ID of the Agent whose state is being recorded
            state: The state of the Agent
        """
        if agent_id not in self._agents:
            self._agents[agent_id] = state
        else:
            logger.warning(f"Agent {agent_id} already in Frame. Adding state skipped!")


class Episode:
    """ An episode that is represented with a collection of Agents and their corresponding frames. """

    def __init__(self, config: EpisodeConfig, metadata: EpisodeMetadata,
                 agents: Dict[int, TrajectoryAgent], frames: List[Frame]):
        self.config = config
        self.metadata = metadata
        self.agents = agents
        self.frames = frames

    def __repr__(self):
        return f"Episode {self.config.recording_id}; {len(self.agents)} agents; {len(self.frames)} frames"

    def __iter__(self):
        self.t = 0
        return self

    def __next__(self):
        if self.t < len(self.frames):
            frame = self.frames[self.t]
            self.t += 1
            return frame
        else:
            raise StopIteration


class IndEpisodeLoader(EpisodeLoader):
    def load(self, config: EpisodeConfig, road_map: Map = None,
             agent_types: List[str] = None, scale: float = None):
        track_file = os.path.join(self.scenario_config.data_root,
                                  '{}_tracks.csv'.format(config.recording_id))
        static_tracks_file = os.path.join(self.scenario_config.data_root,
                                          '{}_tracksMeta.csv'.format(config.recording_id))
        recordings_meta_file = os.path.join(self.scenario_config.data_root,
                                            '{}_recordingMeta.csv'.format(config.recording_id))
        tracks, static_info, meta_info = self._read_from_csv(
            track_file, static_tracks_file, recordings_meta_file)

        num_frames = round(meta_info['frameRate'] * meta_info['duration']) + 1

        agents = {}
        frames = [Frame(i) for i in range(num_frames)]

        for track_meta in static_info:
            agent_meta = self._agent_meta_from_track_meta(track_meta)
            if agent_types is not None and agent_meta.agent_type not in agent_types:
                continue

            agent_id = track_meta['trackId']
            agent_meta = AgentMetadata.interleave(agent_meta, AgentMetadata.CAR_DEFAULT)
            trajectory = StateTrajectory(meta_info["frameRate"])
            track = tracks[agent_id]
            num_agent_frames = int(agent_meta.final_time - agent_meta.initial_time) + 1

            for idx in range(num_agent_frames):
                state = self._state_from_tracks(track, idx, scale, agent_meta)
                trajectory.add_state(state, reload_path=False)
                frames[int(state.time)].add_agent_state(agent_id, state)

            trajectory.calculate_path_and_velocity()
            agent = TrajectoryAgent(agent_id, trajectory.states[0], agent_meta,
                                    fps=meta_info["frameRate"], open_loop=True)
            agent.set_trajectory(trajectory)
            agents[agent_id] = agent

        return Episode(config, EpisodeMetadata(meta_info), agents, frames)

    @staticmethod
    def _state_from_tracks(track, idx, scale: float = None, metadata: AgentMetadata = None):
        time = track['frame'][idx]
        heading = np.deg2rad(track['heading'][idx])
        heading = np.unwrap([0, heading])[1]
        position = np.array([track['xCenter'][idx], track['yCenter'][idx]]) * (scale if scale else 1)
        velocity = np.array([track['xVelocity'][idx], track['yVelocity'][idx]])
        acceleration = np.array([track['xAcceleration'][idx], track['yAcceleration'][idx]])

        return AgentState(time, position, velocity, acceleration, heading, metadata)

    @staticmethod
    def _agent_meta_from_track_meta(track_meta):
        return AgentMetadata(width=track_meta['width'],
                             length=track_meta['length'],
                             agent_type=track_meta['class'],
                             initial_time=track_meta['initialFrame'],
                             final_time=track_meta['finalFrame'])

    def _read_all_recordings_from_csv(self, base_path: str):
        """ This methods reads the tracks and meta information for all recordings given the path of the inD data set.

        Args:
            base_path: Directory containing all csv files of the inD data set
        Returns:
            Tuple of tracks, static track info and recording meta info
        """
        tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
        static_tracks_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
        recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

        all_tracks = []
        all_static_info = []
        all_meta_info = []
        for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                                       static_tracks_files,
                                                                       recording_meta_files):
            logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)
            tracks, static_info, meta_info = self._read_from_csv(track_file, static_tracks_file, recording_meta_file)
            all_tracks.extend(tracks)
            all_static_info.extend(static_info)
            all_meta_info.extend(meta_info)

        return all_tracks, all_static_info, all_meta_info

    def _read_from_csv(self, track_file, static_tracks_file, recordings_meta_file):
        """ This method reads tracks including meta data for a single recording from csv files.

        Args:
            track_file: The input path for the tracks csv file.
            static_tracks_file: The input path for the static tracks csv file.
            recordings_meta_file: The input path for the recording meta csv file.

        Returns:
            Tuple of tracks, static track info and recording info
        """
        static_info = self._read_static_info(static_tracks_file)
        meta_info = self._read_meta_info(recordings_meta_file)
        tracks = self._read_tracks(track_file, meta_info)
        return tracks, static_info, meta_info

    def _read_tracks(self, track_file, meta_info):
        # Read the csv file to a pandas data frame
        df = pandas.read_csv(track_file)

        # To extract every track, group the rows by the track id
        raw_tracks = df.groupby(["trackId"], sort=False)
        ortho_px_to_meter = meta_info["orthoPxToMeter"]
        tracks = []
        for track_id, track_rows in raw_tracks:
            track = track_rows.to_dict(orient="list")

            # Convert scalars to single value and lists to numpy arrays
            for key, value in track.items():
                if key in ["trackId", "recordingId"]:
                    track[key] = value[0]
                else:
                    track[key] = np.array(value)

            track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
            track["bbox"] = calculate_multiple_bboxes(track["xCenter"], track["yCenter"],
                                                      track["length"], track["width"],
                                                      np.deg2rad(track["heading"]))

            # Create special version of some values needed for visualization
            track["xCenterVis"] = track["xCenter"] / ortho_px_to_meter
            track["yCenterVis"] = -track["yCenter"] / ortho_px_to_meter
            track["centerVis"] = np.stack([track["xCenter"], -track["yCenter"]], axis=-1) / ortho_px_to_meter
            track["widthVis"] = track["width"] / ortho_px_to_meter
            track["lengthVis"] = track["length"] / ortho_px_to_meter
            track["headingVis"] = track["heading"] * -1
            track["headingVis"][track["headingVis"] < 0] += 360
            track["bboxVis"] = calculate_multiple_bboxes(track["xCenterVis"], track["yCenterVis"],
                                                         track["lengthVis"], track["widthVis"],
                                                         np.deg2rad(track["headingVis"]))

            tracks.append(track)
        return tracks

    def _read_static_info(self, static_tracks_file: str):
        """This method reads the static info file from highD data.

        Args:
            static_tracks_file: the input path for the static csv file.

        Returns:
            The static dictionary - the key is the track_id and the value is the corresponding data for this track
        """
        return pandas.read_csv(static_tracks_file).to_dict(orient="records")

    def _read_meta_info(self, recordings_meta_file: str):
        """ This method reads the recording info file from ind data.

        Args:
            recordings_meta_file: the path for the recording meta csv file.

        Returns:
            The meta dictionary
        """
        return pandas.read_csv(recordings_meta_file).to_dict(orient="records")[0]
