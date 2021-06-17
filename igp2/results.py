import numpy as np
from typing import List, Dict, Tuple

from igp2.recognition.goalrecognition import GoalsProbabilities
from igp2.data.episode import EpisodeMetadata

class AgentResult:

    def __init__(self, true_goal: int, datum : Tuple[int, GoalsProbabilities, float, np.ndarray] = None):
        if datum is not None: self.data = [datum]
        else: self.data = []

        self.true_goal = true_goal

    def add_data(self, datum : Tuple[int, GoalsProbabilities, float, np.ndarray]):
        self.data.append(datum)

    @property
    def true_goal_probability(self) -> np.ndarray:
        arr = []
        for datum in self.data:
            true_goal_probability = list(datum[1].goals_probabilities.values())[self.true_goal]
            arr.append(true_goal_probability)
        return np.array(arr)

    @property
    def goal_accuracy(self) -> np.ndarray:
        arr = []
        for datum in self.data:
            goal_probs = np.nan_to_num(list(datum[1].goals_probabilities.values()), posinf=0., neginf=0.)
            goal_accuracy = (goal_probs[self.true_goal] == goal_probs.max() and np.count_nonzero(goal_probs == goal_probs.max()) == 1)
            arr.append(goal_accuracy)
        return np.array(arr)

    @property
    def zero_probability(self) -> np.ndarray:
        arr = []
        for datum in self.data:
            goal_probs = np.nan_to_num(list(datum[1].goals_probabilities.values()), posinf=0., neginf=0.)
            arr.append(goal_probs[self.true_goal] == 0)
        return np.array(arr)

    @property
    def reward_difference(self) -> np.ndarray:
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
        arr = np.array([datum[2] for datum in self.data])
        return arr.mean()

    @property
    def position(self) -> np.ndarray:
        return np.array([datum[3] for datum in self.data])


            
class EpisodeResult:

    def __init__(self, metadata: EpisodeMetadata, id : int,  cost_factors : Dict[str, float], datum : Tuple[int, AgentResult] = None):
        self.cost_factors = cost_factors
        self.metadata = metadata
        self.id = id
        
        if datum is not None: self.data = [datum]
        else: self.data = []

    def add_data(self, datum : Tuple[int, AgentResult]):
        self.data.append(datum)

    @property
    def true_goal_probability(self) -> np.ndarray:
        arr = np.array([datum[1].true_goal_probability for datum in self.data])
        arr = np.nan_to_num(arr, posinf=0., neginf=0.)
        return np.mean(arr, axis = 0)

    @property
    def true_goal_ste(self) -> np.ndarray:
        arr = np.array([datum[1].true_goal_probability for datum in self.data])
        arr = np.nan_to_num(arr, posinf=0., neginf=0.)
        return np.std(arr, axis = 0) / np.sqrt(len(self.data))

    @property
    def goal_accuracy(self) -> np.ndarray:
        arr = np.array([datum[1].goal_accuracy for datum in self.data])
        return np.mean(arr, axis = 0)

    @property
    def goal_accuracy_ste(self) -> np.ndarray:
        arr = np.array([datum[1].goal_accuracy for datum in self.data])
        return np.std(arr, axis = 0) / np.sqrt(len(self.data))

    @property
    def zero_probability(self) -> np.ndarray:
        arr = np.array([datum[1].zero_probability for datum in self.data])
        return np.mean(arr, axis = 0)

    @property
    def reward_difference(self) -> np.ndarray:
        arr = np.array([datum[1].reward_difference for datum in self.data])
        return np.nanmean(arr, axis = 0)

    @property
    def reward_difference_std(self) -> np.ndarray:
        arr = np.array([datum[1].reward_difference for datum in self.data])
        return np.nanstd(arr, axis = 0)

    @property
    def reward_difference_median(self) -> np.ndarray:
        arr = np.array([datum[1].reward_difference for datum in self.data])
        return np.nanmedian(arr, axis = 0)

    @property
    def inference_time(self) -> float:
        arr = np.array([datum[1].inference_time for datum in self.data])
        return arr.mean()

class ExperimentResult:

    def __init__(self, datum : Tuple[int, EpisodeResult] = None):
        
        if datum is not None: self.data = [datum]
        else: self.data = []

    def add_data(self, datum : Tuple[int, EpisodeResult]):
        self.data.append(datum)

    @property
    def true_goal_probability(self) -> np.ndarray:
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].true_goal_probability))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].true_goal_probability * num_agents
        return arr / total_agents

    @property
    def goal_accuracy(self) -> np.ndarray:
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].goal_accuracy))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].goal_accuracy * num_agents
        return arr / total_agents

    @property
    def zero_probability(self) -> np.ndarray:
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].zero_probability))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].zero_probability * num_agents
        return arr / total_agents

    @property
    def reward_difference(self) -> np.ndarray:
        total_agents = 0
        arr = np.zeros(len(self.data[0][1].reward_difference))
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            arr += datum[1].reward_difference * num_agents
        return arr / total_agents

    @property
    def inference_time(self) -> float:
        total_agents = 0
        t = 0.
        for datum in self.data:
            num_agents = len(datum[1].data)
            total_agents += num_agents
            t += datum[1].inference_time * num_agents
        return t / total_agents

    @property
    def inference_time_ste(self) -> float:
        arr = []
        for ep_datum in self.data:
            for agent_datum in ep_datum[1].data:
                for frame_datum in agent_datum[1].data:
                    arr.append(frame_datum[2])
        arr = np.array(arr)
        return np.std(arr) / np.sqrt(len(arr))