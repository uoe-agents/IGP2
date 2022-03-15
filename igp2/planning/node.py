from collections import defaultdict
from typing import Dict, List, Tuple

import igp2 as ip
import copy
import logging
import numpy as np

from igp2.planning.mctsaction import MCTSAction

logger = logging.getLogger(__name__)


class Node:
    """ Represents a search node in the MCTS tree. Stores all relevant information for computation of Q-values
    and action selection. Keys must be hashable.

    During search, a Node must be expanded before it can be added to a Tree. Children of the node are stored
    in a dictionary with the key being the state and the value the child node itself.
    """

    def __init__(self, key: Tuple, state: Dict[int, ip.AgentState], actions: List[MCTSAction]):
        if key is None or not isinstance(key, Tuple):
            raise TypeError(f"Node key must not be a tuple.")

        self._key = key
        self._state = state
        self._actions = actions
        self._children = {}

        self._state_visits = 0
        self._q_values = None
        self._action_visits = None

        self._run_results = []
        self._reward_results = defaultdict(list)

    def __repr__(self):
        return str(self.key)

    def expand(self):
        if self._actions is None:
            raise TypeError("Cannot expand node without actions")
        self._q_values = np.zeros(len(self._actions))
        self._action_visits = np.zeros(len(self._actions), dtype=np.int32)

    def add_child(self, child: "Node"):
        """ Add a new child to the dictionary of children. """
        self._children[child.key] = child

    def add_run_result(self, run_result: ip.RunResult):
        """ Add a new simulation run result to the node. """
        self._run_results.append(run_result)

    def add_reward_result(self, reward_results: ip.RewardResult):
        """ Add a new reward outcome to the node if the search has ended here. """
        key = reward_results.node_key[-1]
        assert key in self.actions_names, f"Action {key} not in Node {self._key}"
        self._reward_results[key].append(reward_results)

    def store_q_values(self):
        """ Save the current q_values into the last element of run_results. """
        if self._run_results:
            self._run_results[-1].q_values = copy.copy(self.q_values)

    @property
    def q_values(self) -> np.ndarray:
        """ Return the Q-values corresponding to each action. """
        return self._q_values

    @q_values.setter
    def q_values(self, value: np.ndarray):
        self._q_values = value

    @property
    def key(self) -> Tuple:
        """ Unique hashable key identifying the node and the sequence of actions that lead to it. """
        return self._key

    @property
    def state(self) -> Dict[int, ip.AgentState]:
        """ Return the state corresponding to this node. """
        return self._state

    @property
    def actions(self) -> List[MCTSAction]:
        """ Return possible actions in state of node. """
        return self._actions

    @property
    def actions_names(self) -> List[str]:
        """ Return the human readable names of actions in the node. """
        return [action.__repr__() for action in self._actions]

    @property
    def state_visits(self) -> int:
        """ Return number of time this state has been selected. """
        return self._state_visits

    @state_visits.setter
    def state_visits(self, value: int):
        self._state_visits = value

    @property
    def action_visits(self) -> np.ndarray:
        """ Return number of time each action has been selected in this node. """
        return self._action_visits

    @property
    def children(self) -> Dict[Tuple, "Node"]:
        """ Return the dictionary of children. """
        return self._children

    @property
    def is_leaf(self) -> bool:
        """ Return true if the node has no children. """
        return len(self._children) == 0

    @property
    def run_results(self) -> List[ip.RunResult]:
        """ Return a list of the simulated runs results for this node. """
        return self._run_results

    @property
    def reward_results(self) -> Dict[str, List[ip.RewardResult]]:
        """ Returns a dictionary of reward outcomes where the keys are all possible actions in the node. """
        return self._reward_results

    @property
    def descendants(self):
        """ Return all descendants of this node. """
        descendants = []
        for key, child in self.children.items():
            descendants.append((key, child))
            descendants.extend(child.descendants)
        return descendants
