import igp2 as ip
import copy
from typing import Dict, List, Tuple
import numpy as np


class Node:
    """ Represents a search node in the MCTS tree. Stores all relevant information for computation of Q-values
    and action selection. Keys must be hashable.

    During search, a Node must be expanded before it can be added to a Tree. Children of the node are stored
    in a dictionary with the key being the state and the value the child node itself.
    """

    def __init__(self, key: Tuple, state: Dict[int, ip.AgentState], actions: List[ip.MacroAction]):
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

    def expand(self):
        if self._actions is None:
            raise TypeError("Cannot expand node without actions")
        self._q_values = np.zeros(len(self._actions))
        self._action_visits = np.zeros(len(self._actions), dtype=np.int32)

    def add_child(self, child: "Node"):
        """ Add a new child to the dictionary of children. """
        self._children[child.key] = child

    def add_run_result(self, run_result: ip.RunResult):
        self._run_results.append(run_result)

    def store_q_values(self):
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
    def actions(self) -> List[ip.MacroAction]:
        """ Return possible actions in state of node. """
        return self._actions

    @property
    def actions_names(self) -> List[str]:
        """ Return the human readable names of actions in the node. """
        return [action.__name__ for action in self._actions]

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
