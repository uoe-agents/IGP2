from dataclasses import dataclass
from numbers import Number

import igp2 as ip
import numpy as np
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt
from gui.tracks_import import calculate_rotated_bboxes


class AgentResult:
    """This class will store GoalsProbabilities objects containing goal 
    prediction results belonging to a specific agent"""

    def __init__(self, true_goal: int, datum: Tuple[int, ip.GoalsProbabilities,
                                                    float, np.ndarray] = None):
        """Initialises the class, specifying the index of associated to the true goal and optionally
        adding the first data point, in the form of the tuple
        (frame_id, GoalsProbabilities object, inference time, current position)"""
        if datum is not None:
            self.data = [datum]
        else:
            self.data = []

        self.true_goal = true_goal

    def add_data(self, datum: Tuple[int, ip.GoalsProbabilities, float, np.ndarray]):
        """Adds a new data point, in the form of the tuple 
        (frame_id, GoalsProbabilities object, inference time, current position)"""
        self.data.append(datum)

    @property
    def true_goal_probability(self) -> np.ndarray:
        """Returns the probabilities of the true goal for each data point."""
        arr = []
        for datum in self.data:
            true_goal_probability = list(datum[1].goals_probabilities.values())[self.true_goal]
            arr.append(true_goal_probability)
        return np.array(arr)

    @property
    def goal_accuracy(self) -> np.ndarray:
        """Returns True if the true goal is the most likely, false otherwise, 
        for each data point."""
        arr = []
        for datum in self.data:
            goal_probs = np.nan_to_num(list(datum[1].goals_probabilities.values()),
                                       posinf=0., neginf=0.)
            goal_accuracy = (goal_probs[self.true_goal] == goal_probs.max() and np.count_nonzero(
                goal_probs == goal_probs.max()) == 1)
            arr.append(goal_accuracy)
        return np.array(arr)

    @property
    def zero_probability(self) -> np.ndarray:
        """Returns true if the true goal has a zero probability 
        (considered unfeasible), false otherwise, for each data point."""
        arr = []
        for datum in self.data:
            goal_probs = np.nan_to_num(list(datum[1].goals_probabilities.values()), posinf=0., neginf=0.)
            arr.append(goal_probs[self.true_goal] == 0)
        return np.array(arr)

    @property
    def reward_difference(self) -> np.ndarray:
        """Returns reward difference associated with the true goal, 
        for each data point."""
        # check if reward difference data is available
        arr = []
        for datum in self.data:
            reward_difference = list(datum[1].reward_difference.values())[self.true_goal]
            if reward_difference is None:
                arr.append(np.NaN)
            else:
                arr.append(reward_difference)

        # if data is unavailable, try to compute manually
        if np.isnan(arr).all():
            arr = []
            for datum in self.data:
                optimum_reward = list(datum[1].optimum_reward.values())[self.true_goal]
                current_reward = list(datum[1].current_reward.values())[self.true_goal]
                if current_reward is None or optimum_reward is None:
                    arr.append(np.NaN)
                else:
                    arr.append(current_reward - optimum_reward)

        return np.array(arr)

    @property
    def inference_time(self) -> float:
        """Returns inference time for each data point."""
        arr = np.array([datum[2] for datum in self.data])
        return arr.mean()

    @property
    def position(self) -> np.ndarray:
        """Returns agent current position for each data point."""
        return np.array([datum[3] for datum in self.data])


class EpisodeResult:
    """This class stores result for an entire episode, where each data point
     contains an AgentResult object"""

    def __init__(self, metadata: ip.data.EpisodeMetadata, id: int, cost_factors: Dict[str, float],
                 datum: Tuple[int, AgentResult] = None):
        """Initialises the class, storing the episode metadata, cost factors
        and optionally add a data point in the form of the tuple 
        (agent_id, AgentResult object)"""
        self.cost_factors = cost_factors
        self.metadata = metadata
        self.id = id

        if datum is not None:
            self.data = [datum]
        else:
            self.data = []

    def add_data(self, datum: Tuple[int, AgentResult]):
        """Adds a new data point, in the form of the tuple 
        (agent_id, AgentResult object)"""
        self.data.append(datum)

    @property
    def true_goal_probability(self) -> np.ndarray:
        """The mean true goal probability across the episode."""
        arr = np.array([datum[1].true_goal_probability for datum in self.data])
        arr = np.nan_to_num(arr, posinf=0., neginf=0.)
        return np.mean(arr, axis=0)

    @property
    def true_goal_ste(self) -> np.ndarray:
        """The standard error of the true goal probability across the episode."""
        arr = np.array([datum[1].true_goal_probability for datum in self.data])
        arr = np.nan_to_num(arr, posinf=0., neginf=0.)
        return np.std(arr, axis=0) / np.sqrt(len(self.data))

    @property
    def goal_accuracy(self) -> np.ndarray:
        """The accuracy across the episode, defined as the fraction of how many
        times the true goal was most likely, over the total number of data points."""
        arr = np.array([datum[1].goal_accuracy for datum in self.data])
        return np.mean(arr, axis=0)

    @property
    def goal_accuracy_ste(self) -> np.ndarray:
        """The standard error of the goal accuracy."""
        arr = np.array([datum[1].goal_accuracy for datum in self.data])
        return np.std(arr, axis=0) / np.sqrt(len(self.data))

    @property
    def zero_probability(self) -> np.ndarray:
        """The fraction of times the true goal was considered unfeasible, 
        over the total number of data points."""
        arr = np.array([datum[1].zero_probability for datum in self.data])
        return np.mean(arr, axis=0)

    @property
    def reward_difference(self) -> np.ndarray:
        """The average reward difference across the episode."""
        arr = np.array([datum[1].reward_difference for datum in self.data])
        return np.nanmean(arr, axis=0)

    @property
    def reward_difference_std(self) -> np.ndarray:
        """The standard deviation associated with the reward difference."""
        arr = np.array([datum[1].reward_difference for datum in self.data])
        return np.nanstd(arr, axis=0)

    @property
    def reward_difference_median(self) -> np.ndarray:
        """The median reward difference."""
        arr = np.array([datum[1].reward_difference for datum in self.data])
        return np.nanmedian(arr, axis=0)

    @property
    def inference_time(self) -> float:
        """The average inference time accross the episode."""
        arr = np.array([datum[1].inference_time for datum in self.data])
        return arr.mean()


class ExperimentResult:
    """This class stores results for an entire experiment ran across 
    multiple scenarios, each data point contains an EpisodeResult object"""

    def __init__(self, datum: Tuple[int, EpisodeResult] = None):
        """Initialises class and optionally adds a data point, in the 
        form of the tuple (recording_id, EpisodeResult object) """
        if datum is not None:
            self.data = [datum]
        else:
            self.data = []

    def add_data(self, datum: Tuple[int, EpisodeResult]):
        """Adds a data point, in the form of the tuple 
        (recording_id, EpisodeResult object)"""
        self.data.append(datum)

    @property
    def true_goal_probability(self) -> np.ndarray:
        """Calculates the average true goal probability across all agents 
        evaluated in the experiment."""
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].true_goal_probability))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].true_goal_probability * num_agents
        return arr / total_agents

    @property
    def goal_accuracy(self) -> np.ndarray:
        """Calculates the average goal accuracy across all agents evaluated in 
        the experiment."""
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].goal_accuracy))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].goal_accuracy * num_agents
        return arr / total_agents

    @property
    def zero_probability(self) -> np.ndarray:
        """Calculates the average zero true goal probability across all agents 
        evaluated in the experiment."""
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].zero_probability))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].zero_probability * num_agents
        return arr / total_agents

    @property
    def reward_difference(self) -> np.ndarray:
        """Calculates the average reward difference across all agents evaluated 
        in the experiment."""
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].reward_difference))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].reward_difference * num_agents
        return arr / total_agents

    @property
    def inference_time(self) -> float:
        """Calculates the average inference_time across all agents evaluated in 
        the experiment."""
        total_agents = 0
        t = 0.
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            t += datum[1].inference_time * num_agents
        return t / total_agents

    @property
    def inference_time_ste(self) -> float:
        """Calculates the standard error of the inference time across all agents
         evaluated in the experiment."""
        arr = []
        for ep_datum in self.data:
            for agent_datum in ep_datum[1].data:
                for frame_datum in agent_datum[1].data:
                    arr.append(frame_datum[2])
        arr = np.array(arr)
        return np.std(arr) / np.sqrt(len(arr))


@dataclass
class RunResult:
    """ Class storing results of the simulated rollout in MCTS. """
    agents: Dict[int, ip.Agent]
    ego_id: int
    ego_trajectory: ip.Trajectory
    collided_agents_ids: List[int]
    goal_reached: bool
    q_values: np.ndarray = None

    @property
    def ego_macro_action(self) -> str:
        return self.agents[self.ego_id].current_macro.__repr__()

    @property
    def ego_maneuvers(self) -> List[str]:
        maneuvers = self.agents[self.ego_id].current_macro.maneuvers
        return [man.__repr__() for man in maneuvers]

    @property
    def ego_maneuvers_trajectories(self) -> List[ip.VelocityTrajectory]:
        """This returns the generated open loop trajectories for each maneuver."""
        maneuvers = self.agents[self.ego_id].current_macro.maneuvers
        return [man.trajectory for man in maneuvers]

    def plot(self, t: int, scenario_map: ip.Map, axis: plt.Axes = None) -> plt.Axes:
        """ Plot the current agents and the road layout at timestep t for visualisation purposes.
        Plots the OPEN LOOP trajectories that the agents will attempt to follow, alongside a colormap
        representing velocities.

        Args:
            t: timestep at which the plot should be generated
            scenario_map: The road layout
            axis: Axis to draw on
        """
        if axis is None:
            fig, axis = plt.subplots()

        color_map_ego = plt.cm.get_cmap('Reds')
        color_map_non_ego = plt.cm.get_cmap('Blues')
        color_ego = 'r'
        color_non_ego = 'b'
        color_bar_non_ego = None

        ip.plot_map(scenario_map, markings=True, ax=axis)
        for agent_id, agent in self.agents.items():

            if isinstance(agent, ip.MacroAgent):
                color = color_ego
                color_map = color_map_ego
                path = []
                velocity = []
                for man in agent.current_macro.maneuvers:
                    path.extend(man.trajectory.path)
                    velocity.extend(man.trajectory.velocity)
                path = np.array(path)
                velocity = np.array(velocity)
            elif isinstance(agent, ip.TrajectoryAgent):
                color = color_non_ego
                color_map = color_map_non_ego
                path = agent.trajectory.path
                velocity = agent.trajectory.velocity

            vehicle = agent.vehicle
            bounding_box = calculate_rotated_bboxes(agent.trajectory_cl.path[t][0], agent.trajectory_cl.path[t][1],
                                                    vehicle.length, vehicle.width,
                                                    agent.trajectory_cl.heading[t])
            pol = plt.Polygon(bounding_box[0], color=color)
            axis.add_patch(pol)
            agent_plot = axis.scatter(path[:, 0], path[:, 1], c=velocity, cmap=color_map, vmin=-4, vmax=20, s=8)
            if isinstance(agent, ip.MacroAgent):
                plt.colorbar(agent_plot)
                plt.text(0, 0.1, 'Current Macro Action: ' + self.ego_macro_action, horizontalalignment='left',
                         verticalalignment='bottom', transform=axis.transAxes)
                plt.text(0, 0.05, 'Maneuvers: ' + str(self.ego_maneuvers),
                         horizontalalignment='left', verticalalignment='bottom', transform=axis.transAxes)
                if len(self.ego_maneuvers) > 1:
                    maneuver_end_idx = agent.maneuver_end_idx
                    if len(maneuver_end_idx) < len(self.ego_maneuvers):
                        maneuver_end_idx.append(len(agent.trajectory_cl.states))
                    current_maneuver_id = np.min(np.nonzero(np.array(maneuver_end_idx) > t))
                    plt.text(0, 0.0, 'Current Maneuver: ' + self.ego_maneuvers[current_maneuver_id],
                             horizontalalignment='left', verticalalignment='bottom', transform=axis.transAxes)
            elif isinstance(agent, ip.TrajectoryAgent) and color_bar_non_ego is None:
                color_bar_non_ego = plt.colorbar(agent_plot)
            plt.text(*agent.trajectory_cl.path[t], agent_id)
        return axis


@dataclass
class RewardResult:
    """ Class to store reward outcomes from the MCTS search. """
    cost: ip.Cost = None
    collision: float = None
    termination: float = None
    death: float = None

    @property
    def total_reward(self):
        """ Calculate the total reward stored in the class"""
        return sum([v for v in vars(self).values()])


class MCTSResultTemplate:
    pass


class MCTSResult(MCTSResultTemplate):

    def __init__(self, tree: "ip.Tree" = None, samples: dict = None):
        self.tree = tree
        self.samples = samples

    def plot_q_values(self, key: Tuple, axis: plt.Axes = None) -> plt.Axes:

        if axis is None:
            fig, axis = plt.subplots()

        node = self.tree.tree[key]
        all_q = np.empty((len(node.run_results), len(node.actions)), float)
        for i, run_result in enumerate(node.run_results):
            all_q[i, :] = run_result.q_values

        for i in range(0, len(node.actions)):
            label = "Macro Action: " + node.actions_names[i]
            plt.plot(all_q[:, i], label=label)

        axis.set(ylabel="Q Value", xlabel="Run number")
        axis.legend()

        return axis


class AllMCTSResult(MCTSResultTemplate):

    def __init__(self, mcts_result: MCTSResult = None):
        if mcts_result is None:
            self.mcts_results = []
        else:
            self.mcts_results = [mcts_result]

    def add_data(self, mcts_result: MCTSResult):
        self.mcts_results.append(mcts_result)


class PlanningResult:

    def __init__(self, scenario_map: ip.Map, mcts_result: MCTSResultTemplate = None, timestep: float = None,
                 frame: Dict[int, ip.AgentState] = None, goals_probabilities: ip.GoalsProbabilities = None):

        self.scenario_map = scenario_map

        if mcts_result is None:
            self.results = []
        else:
            self.results = [mcts_result]

        if timestep is None:
            self.timesteps = []
        else:
            self.timesteps = [timestep]

        if frame is None:
            self.frames = []
        else:
            self.frames = [frame]

        if goals_probabilities is None:
            self.goal_probabilities = []
        else:
            self.goal_probabilities = [goals_probabilities]

    def add_data(self, mcts_result: MCTSResultTemplate = None, timestep: float = None, frame: Dict[
        int, ip.AgentState] = None, goals_probabilities: ip.GoalsProbabilities = None):

        self.results.append(mcts_result)
        self.timesteps.append(timestep)
        self.frames.append(frame)
        self.goal_probabilities.append(goals_probabilities)
