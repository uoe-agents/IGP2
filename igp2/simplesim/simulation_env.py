import logging
import random
from copy import deepcopy
from typing import Dict, List, Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from shapely import Polygon
import matplotlib.pyplot as plt

from igp2.agents.mcts_agent import MCTSAgent
from igp2.agents.traffic_agent import TrafficAgent
from igp2.core.goal import BoxGoal
from igp2.core import util
from igp2.core.agentstate import AgentState
from igp2.simplesim.simulation import Simulation
from igp2.opendrive import Map
from igp2.core.agentstate import AgentMetadata
from igp2.planlibrary.maneuver import Maneuver
from igp2.planlibrary.macro_action import (
    MacroActionFactory,
    MacroActionConfig,
    MacroAction,
)
from igp2.simplesim.plot_simulation import plot_simulation
from igp2.core.config import Configuration


logger = logging.getLogger(__name__)
MAX_ITERS = 10000


class SimulationEnv(gym.Env):
    """A gym environment wrapper around the Simulation class."""

    metadata = {"render_modes": ["human", "plot"]}

    def __init__(self, config: Dict[str, Any], render_mode: str = None, max_iters: int = MAX_ITERS):
        """Initialise new simple simulation environment as a ParallelEnv.
        Args:
            config: Scenario configuration object.
            open_loop: If true then no physical controller will be applied.
        """
        self.config = config
        self.max_iters = max_iters

        # Set IGP2 configs
        ip_config = Configuration()
        ip_config.set_properties(**config["scenario"])

        # Initialize simulation
        self.scenario_map = Map.parse_from_opendrive(config["scenario"]["map_path"])
        self.fps = int(config["scenario"]["fps"]) if "fps" in config["scenario"] else 20
        self.open_loop = config["scenario"].get("open_loop", False)
        self.separate_ego = config["scenario"].get("separate_ego", False)
        self._simulation = Simulation(self.scenario_map, self.fps, self.open_loop)

        # Set up Env variables
        self.n_agents = None
        self.reset_observation_space(init=True)
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
        )
        self.render_mode = render_mode

    def reset_observation_space(self, last_obs: dict = None, init: bool = False):
        """Reset the observation space to default values."""
        if init:
            n_agents = len(self.config["agents"])
        elif last_obs is not None:
            n_agents = len(last_obs["position"])
        else:
            n_agents = len(self.simulation.agents)

        if self.n_agents == n_agents:
            return
        self.n_agents = n_agents

        self.observation_space = gym.spaces.Dict(
            position=Box(
                low=-np.inf, high=np.inf, shape=(self.n_agents, 2), dtype=np.float64
            ),
            velocity=Box(
                low=-np.inf, high=np.inf, shape=(self.n_agents, 2), dtype=np.float64
            ),
            acceleration=Box(
                low=-np.inf, high=np.inf, shape=(self.n_agents, 2), dtype=np.float64
            ),
            heading=Box(
                low=-np.inf, high=np.inf, shape=(self.n_agents,), dtype=np.float64
            ),
        )

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        elif self.render_mode == "human":
            string = f"Step {self._simulation.t}:\n"
            for agent_id, state in self._simulation.state.items():
                string += (
                    f"  Agent {agent_id} - "
                    f"Pos: {np.round(state.position, 2)} - "
                    f"Vel: {np.round(state.speed, 2)} - "
                    f"Mcr: {state.macro_action} - "
                    f"Man: {state.maneuver}\n"
                )
            logger.info(string)
        elif (
            self.render_mode == "plot"
            and self._simulation.t % self.config["scenario"].get("plot_freq", 10) == 0
        ):
            plot_simulation(self._simulation, debug=False)
            plt.show()

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: The action to take in the environment.
        """
        _, collisions = self._simulation.take_actions(action)

        ego_agent = self._simulation.agents[0]
        goal_reached = ego_agent.goal.reached(ego_agent.state.position)
        if goal_reached:
            ego_agent.trajectory_cl.calculate_path_and_velocity()
        termination = not ego_agent.alive or goal_reached
        env_truncation = self._simulation.t >= self.max_iters
        observation, info = self._get_obs(return_frame=True)
        self.reset_observation_space(observation)

        reward = ego_agent.reward(collisions[0],
                                  ego_agent.alive,
                                  ego_agent.trajectory_cl,
                                  ego_agent.goal if goal_reached else None,
                                  env_truncation)
        if reward is None:
            reward = 0.0
        else:
            info["reward"] = deepcopy(ego_agent.reward)

        self.render()

        return observation, reward, termination, env_truncation, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state.

        Args:
            seed: Random seed to use for environment reset.
            options: Dictionary of options to pass to the environment.
                - add_agents: Whether to add agents based on the config file.
                                Default is True.
        """
        random.seed(seed)
        super().reset(seed=seed)

        self._simulation.reset()

        add_agents = options.get("add_agents", True) if options else True
        if not add_agents:
            return self._get_obs(return_frame=True)

        ego_agent = None
        initial_frame = self._generate_random_frame(
            self.scenario_map, self.config
        )
        for agent_config in self.config["agents"]:
            agent, rolename = self.create_agent(
                agent_config, self.scenario_map, initial_frame, self.fps,
            )
            self._simulation.add_agent(agent, rolename)
            if rolename == "ego":
                ego_agent = agent

        observation, info = self._get_obs(return_frame=True)
        self.reset_observation_space(observation)

        if self.separate_ego:
            if not ego_agent:
                raise ValueError(
                    "config.scenario.separate_ego was true but no agent "
                    "with rolename == 'ego' found in scenario."
                )
            info["ego"] = ego_agent

        return observation, info

    def close(self):
        """No rendering is performed currently so nothing to close."""
        pass

    def _get_obs(self, return_frame: bool = False) -> Dict[str, np.ndarray]:
        """Convert an AgentState object to a gym observation space."""
        frame = self.simulation.get_observations().frame
        positions = []
        velocities = []
        accelerations = []
        headings = []
        for agent_state in frame.values():
            positions.append(agent_state.position)
            velocities.append(agent_state.velocity)
            accelerations.append(agent_state.acceleration)
            headings.append(agent_state.heading)

        obs = {
            "position": np.array(positions),
            "velocity": np.array(velocities),
            "acceleration": np.array(accelerations),
            "heading": np.array(headings),
        }
        if return_frame:
            return obs, frame
        return obs

    @property
    def simulation(self) -> Simulation:
        """Return the current simulation object."""
        return self._simulation

    @property
    def t(self) -> int:
        """Return the current simulation time."""
        return self._simulation.t

    def create_agent(self, agent_config, scenario_map, frame, fps):
        base_agent = {
            "agent_id": agent_config["id"],
            "initial_state": frame[agent_config["id"]],
            "goal": BoxGoal(util.Box(**agent_config["goal"]["box"])),
            "fps": fps,
        }

        mcts_agent = {
            "scenario_map": scenario_map,
            "cost_factors": agent_config.get("cost_factors", None),
            "view_radius": agent_config.get("view_radius", None),
            "kinematic": True,
            "velocity_smoother": agent_config.get("velocity_smoother", None),
            "goal_recognition": agent_config.get("goal_recognition", None),
            "stop_goals": agent_config.get("stop_goals", False),
        }

        if agent_config["type"] == "MCTSAgent":
            agent = MCTSAgent(**base_agent, **mcts_agent, **agent_config["mcts"])
            rolename = "ego"
        elif agent_config["type"] == "TrafficAgent":
            if "macro_actions" in agent_config and agent_config["macro_actions"]:
                base_agent["macro_actions"] = self._to_ma_list(
                    deepcopy(agent_config["macro_actions"]),
                    agent_config["id"],
                    frame,
                    scenario_map,
                )
            rolename = agent_config.get("rolename", "car")
            agent = TrafficAgent(**base_agent)
        else:
            raise ValueError(f"Unsupported agent type {agent_config['type']}")
        return agent, rolename

    def _generate_random_frame(self, layout: Map, config) -> Dict[int, AgentState]:
        """Generate a new frame with randomised spawns and velocities for each vehicle.

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
            spawn_box = util.Box(**agent["spawn"]["box"])
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
            vel = min(vel, Maneuver.MAX_SPEED)
            spawn_velocity = vel * np.array(
                [np.cos(spawn_heading), np.sin(spawn_heading)]
            )

            agent_metadata = (
                AgentMetadata(**agent["metadata"])
                if "metadata" in agent
                else AgentMetadata(**AgentMetadata.CAR_DEFAULT)
            )

            ret[agent["id"]] = AgentState(
                time=0,
                position=spawn_position,
                velocity=spawn_velocity,
                acceleration=np.array([0.0, 0.0]),
                heading=spawn_heading,
                metadata=agent_metadata,
            )
        return ret

    def _to_ma_list(
        self,
        ma_confs: List[Dict[str, Any]],
        agent_id: int,
        start_frame: Dict[int, AgentState],
        scenario_map: Map,
    ) -> List[MacroAction]:
        """Convert a list of macro action configurations to a list of MacroAction objects."""
        mas = []
        for config in ma_confs:
            config["open_loop"] = False
            frame = start_frame if not mas else mas[-1].final_frame
            if "target_sequence" in config:
                lane_list = [
                    scenario_map.get_lane(rid, lid)
                    for rid, lid in config["target_sequence"]
                ]
                config["target_sequence"] = lane_list
            ma = MacroActionFactory.create(
                MacroActionConfig(config), agent_id, frame, scenario_map
            )
            mas.append(ma)
        return mas
