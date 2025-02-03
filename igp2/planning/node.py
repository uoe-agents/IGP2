import copy
from collections import defaultdict
from typing import Dict, List, Tuple

import logging
import numpy as np

from igp2.planning.mctsaction import MCTSAction
from igp2.planning.reward import Reward
from igp2.core.results import RunResult
from igp2.core.agentstate import AgentState
from igp2.core.util import copy_agents_dict

logger = logging.getLogger(__name__)


class Node:
    """ Represents a search node in the MCTS tree. Stores all relevant information for computation of Q-values
    and action selection. Keys must be hashable.

    During search, a Node must be expanded before it can be added to a Tree. Children of the node are stored
    in a dictionary with the key being the state and the value the child node itself.
    """

    def __init__(self, key: Tuple, state: Dict[int, AgentState], actions: List[MCTSAction]):
        if key is None or not isinstance(key, Tuple):
            raise TypeError(f"Node key must not be a tuple.")

        self._key = key
        self._state = state
        self._actions = actions
        self._children = {}

        self._state_visits = 0
        self._q_values = None
        self._action_visits = None

        self._run_result = None
        self._reward_results = defaultdict(list)

    def __repr__(self):
        return str(self.key)

    def __deepcopy__(self, memodict={}):
        """ Overwrite standard deepcopy to avoid infinite recursion with run results. """
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_run_result" and isinstance(v, RunResult):
                run_result = RunResult.__new__(RunResult)
                memodict[id(run_result)] = run_result
                for rk, rv in self._run_result.__dict__.items():
                    if rk == "agents":
                        setattr(run_result, rk, copy_agents_dict(rv, memodict))
                    else:
                        setattr(run_result, rk, copy.deepcopy(rv, memodict))
                setattr(result, k, run_result)
            else:
                setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def expand(self):
        if self._actions is None:
            raise TypeError("Cannot expand node without actions")
        self._q_values = np.zeros(len(self._actions))
        self._action_visits = np.zeros(len(self._actions), dtype=np.int32)

    def add_child(self, child: "Node"):
        """ Add a new child to the dictionary of children. """
        self._children[child.key] = child

    def add_reward_result(self, key: Tuple[str], reward_results: Reward):
        """ Add a new reward outcome to the node if the search has ended here. """
        action = key[-1]
        assert action in self.actions_names, f"Action {action} not in Node {self._key}"
        self._reward_results[action].append(reward_results)

    def store_q_values(self):
        """ Save the current q_values into the last element of run_results. """
        if self._run_result is not None:
            self._run_result.q_values = copy.copy(self.q_values)

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
    def state(self) -> Dict[int, AgentState]:
        """ Return the state corresponding to this node. """
        return self._state

    @property
    def actions(self) -> List[MCTSAction]:
        """ Return possible actions in state of node. """
        return self._actions

    @property
    def actions_names(self) -> List[str]:
        """ Return the human-readable names of actions in the node. """
        return [str(action) for action in self._actions]

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
    def run_result(self) -> RunResult:
        """ Return a list of the simulated runs results for this node. """
        return self._run_result

    @run_result.setter
    def run_result(self, value: RunResult):
        self._run_result = value

    @property
    def reward_results(self) -> Dict[str, List[Reward]]:
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
