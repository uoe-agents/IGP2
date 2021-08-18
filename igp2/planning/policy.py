import abc
from typing import Any

import numpy as np
from math import sqrt

from igp2.planning.node import Node


class Policy(abc.ABC):
    """ Abstract class for implementing various selection policies """

    def select(self, node: Node) -> Any:
        raise NotImplementedError


class UCB1(Policy):
    """ Policy implementing the UCB1 selection policy.
    Ref: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation"""

    def __init__(self, c: float = sqrt(2)):
        """ Initialise new UCB1 policy

        Args:
            c: the exploitation parameter
        """
        self.c = c

    def select(self, node: Node) -> Any:
        values = node.q_values + self.c * np.sqrt(np.log(node.state_visits + 1) / node.action_visits)
        idx = np.argmax(values)
        return node.actions[idx]

