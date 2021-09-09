import numpy as np
from typing import List, Dict, Tuple

from igp2.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.data.episode import EpisodeMetadata
from igp2.agent import Agent
from igp2.trajectory import VelocityTrajectory


class AgentResult:
    """This class will store GoalsProbabilities objects containing goal 
    prediction results belonging to a specific agent"""

    def __init__(self, true_goal: int, datum: Tuple[int, GoalsProbabilities,
                                                    float, np.ndarray] = None):
        """Initialises the class, specifying the index of associated to the true goal and optionally
        adding the first data point, in the form of the tuple
        (frame_id, GoalsProbabilities object, inference time, current position)"""
        if datum is not None:
            self.data = [datum]
        else:
            self.data = []

        self.true_goal = true_goal

    def add_data(self, datum: Tuple[int, GoalsProbabilities, float, np.ndarray]):
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

    def __init__(self, metadata: EpisodeMetadata, id: int, cost_factors: Dict[str, float],
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


class RunResult:

    def __init__(self, agents: Dict[int, Agent], ego_id: int, ego_trajectory, collided_agents_ids: List[int],
                 goal_reached: bool):
        self.agents = agents
        self.ego_id = ego_id
        self.ego_trajectory = ego_trajectory
        self.collisions = collided_agents_ids
        self.goal_reached = goal_reached
        self.q_values = None

    @property
    def ego_macro_action(self) -> str:
        return self.agents[self.ego_id].current_macro.__repr__()

    @property
    def ego_maneuvers(self) -> List[str]:
        maneuvers = self.agents[self.ego_id].current_macro.maneuvers
        return [man.__repr__() for man in maneuvers]

    @property
    def ego_maneuvers_trajectories(self) -> List[VelocityTrajectory]:
        """This returns the generated open loop trajectories for each maneuver."""
        maneuvers = self.agents[self.ego_id].current_macro.maneuvers
        return [man.trajectory for man in maneuvers]


class MCTSResultTemplate:
    pass


class MCTSResult(MCTSResultTemplate):

    def __init__(self, tree=None):
        self.tree = tree


class AllMCTSResult(MCTSResultTemplate):

    def __init__(self, mcts_result: MCTSResult = None):
        if mcts_result is None:
            self.mcts_results = []
        else:
            self.mcts_results = [mcts_result]

    def add_data(self, mcts_result: MCTSResult):
        self.mcts_results.append(mcts_result)


class PlanningResult:

    def __init__(self, scenario_map: Map, mcts_result: MCTSResultTemplate = None, timestep: float = None,
                 frame: Dict[int, AgentState] = None, goals_probabilities: GoalsProbabilities = None):

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
        int, AgentState] = None, goals_probabilities: GoalsProbabilities = None):

        self.results.append(mcts_result)
        self.timesteps.append(timestep)
        self.frames.append(frame)
        self.goal_probabilities.append(goals_probabilities)
