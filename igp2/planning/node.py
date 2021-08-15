from typing import Hashable, Iterable

import numpy as np


class Node:
    """ Represents a search node in the MCTS tree. Stores all relevant information for computation of Q-values
    and action selection. States must be hashable.

    During search, a Node must be expanded before it can be added to a Tree. Children of the node are stored
    in a dictionary with the key being the state and the value the child node itself.
    """

    def __init__(self, state: Hashable, actions: Iterable):
        if state is None or not isinstance(state, Hashable):
            raise TypeError(f"Node state must not be None and must be hashable")

        self._state = state
        self._actions = actions
        self._children = {}

        self._state_visits = 0
        self._q_values = None
        self._action_visits = None

    def expand(self):
        if self._actions is None:
            raise TypeError("Cannot expand node without actions")
        self._q_values = np.zeros(len(self._actions))
        self._action_visits = np.zeros(len(self._actions), dtype=np.int32)

    def add_child(self, child: "Node"):
        """ Add a new child to the dictionary of children. """
        self._children[child.state] = child

    @property
    def q_values(self) -> np.ndarray:
        """ Return the Q-values corresponding to each action. """
        return self._q_values

    @property
    def state(self) -> Hashable:
        """ Return the state corresponding to this node. """
        return self._state

    @property
    def actions(self) -> Iterable:
        """ Return possible actions in state of node. """
        return self._actions

    @property
    def state_visits(self) -> int:
        """ Return number of time this state has been selected. """
        return self._state_visits

    @property
    def action_visits(self) -> np.ndarray:
        """ Return number of time each action has been selected in this node. """
        return self._action_visits