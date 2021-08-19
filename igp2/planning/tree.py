import logging
import numpy as np
from typing import Dict, Optional, List, Tuple

from igp2.planning.node import Node
from igp2.planning.policy import Policy, UCB1, MaxPolicy

logger = logging.getLogger(__name__)


class Tree:
    """ Defines an abstract tree consisting of nodes.

    A Tree is represented as a dictionary where each key corresponds to a key of the node and the values are
    the corresponding node object. A node is a leaf if it has no children and keys correspond to the action selection
    history that led to the node.
    """

    def __init__(self, root: Node, action_policy: Policy = None, plan_policy: Policy = None):
        """ Initialise a new Tree with the given root.

        Args:
            root: the root node
            action_policy: policy for selecting actions (default: UCB1)
            plan_policy: policy for selecting the final plan (default: Max)
        """
        self._root = root
        self._tree = {root.key: root}

        self._action_policy = action_policy if action_policy is not None else UCB1()
        self._plan_policy = plan_policy if plan_policy is not None else MaxPolicy()

    def __contains__(self, item) -> bool:
        return item in self._tree

    def __getitem__(self, item) -> Optional[Node]:
        if item in self._tree:
            return self._tree[item]
        return None

    def add_node(self, node: Node):
        """ Add a new node to the tree if not already in the tree. """
        if node.state not in self._tree:
            self._tree[node.key] = node
        else:
            logger.warning(f"Node {node.key} already in the tree!")

    def add_child(self, parent: Node, child: Node):
        """ Add a new child to the tree and assign it under an existing parent node. """
        if parent.key in self._tree:
            self.add_node(child)
            self._tree[parent.key].add_child(child)
        else:
            logger.warning(f"Parent {parent.key} not in the tree!")

    def select_action(self, node: Node):
        """ Select one of the actions in the node using the specified policy and update node statistics """
        action, idx = self._action_policy.select(node)
        node.action_visits[idx] += 1
        return action

    def select_plan(self) -> List:
        """ Return the best sequence of actions from the root according to the specified policy. """
        plan = []
        node = self.root
        while node is not None and node.state_visits > 0:
            next_action = self._plan_policy.select(node)
            plan.append(next_action)
            node = self[tuple(list(node.key) + [next_action.__name__])]
        return plan

    def backprop(self, r: float, final_key: Tuple):
        """ Back-propagate the reward through the search branches.

        Args:
            r: reward at end of simulation
            final_key: final node key including final action
        """
        key = final_key
        while key != self._root.key:
            node, action, child = (self[key[:-1]], key[-1], self[key])

            idx = [a.__name__ for a in node.actions].index(action)
            action_visit = node.action_visits[idx]

            # Eq. 8 - back-propagation rule
            q = r if node.is_leaf else np.max(child.q_values)
            node.q_values[idx] += (q - node.q_values[idx]) / action_visit

            key = key[:-1]

    @property
    def root(self) -> Node:
        """ Return the root node of the tree"""
        return self._root

    @property
    def tree(self) -> Dict:
        return self._tree