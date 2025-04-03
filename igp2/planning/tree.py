import logging
import numpy as np
from typing import Dict, Optional, List, Tuple

from igp2.recognition.goalprobabilities import GoalWithType, GoalsProbabilities
from igp2.core.trajectory import VelocityTrajectory
from igp2.planning.node import Node
from igp2.planning.policy import Policy, UCB1, MaxPolicy
from igp2.planning.mctsaction import MCTSAction

logger = logging.getLogger(__name__)


class Tree:
    """ Defines an abstract tree consisting of nodes.

    A Tree is represented as a dictionary where each key corresponds to a key of the node and the values are
    the corresponding node object. A node is a leaf if it has no children and keys correspond to the action selection
    history that led to the node.
    """

    def __init__(self,
                 root: Node,
                 action_policy: Policy = None,
                 plan_policy: Policy = None):
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

    def _add_node(self, node: Node):
        """ Add a new node to the tree if not already in the tree. """
        if node.key not in self._tree:
            self._tree[node.key] = node
        else:
            logger.warning(f"Node {node.key} already in the tree!")

    def add_child(self, parent: Node, child: Node):
        """ Add a new child to the tree and assign it under an existing parent node. """
        if parent.key in self._tree:
            self._add_node(child)
            if child not in parent.children:
                parent.add_child(child)
            else:
                logger.warning(f"    Child {child.key} already in the parent {parent.key}!")
        else:
            raise RuntimeError(f"Parent {parent.key} not in the tree!")

    def select_action(self, node: Node) -> MCTSAction:
        """ Select one of the actions in the node using the specified policy and update node statistics """
        action, idx = self._action_policy.select(node)
        node.action_visits[idx] += 1
        return action

    def select_plan(self) -> Tuple[List, Tuple[str]]:
        """ Return the best sequence of actions from the root according to the specified policy. """
        plan = []
        node = self.root
        next_key = self.root.key

        while node is not None and node.state_visits > 0:
            next_action, action_idx = self._plan_policy.select(node)
            plan.append(next_action)
            next_key = tuple(list(node.key) + [str(next_action)])
            node = self[next_key]

        return plan, next_key

    def backprop(self, r: float, final_key: Tuple, force_reward: bool = False):
        """ Back-propagate the reward through the search branches.

        Args:
            r: reward at end of simulation
            final_key: final node key including final action
            force_reward: if true, then use the reward for a non-terminal state. (default: False)
        """
        key = final_key
        while key != self._root.key:
            node, action, child = (self[key[:-1]], key[-1], self[key])

            idx = node.actions_names.index(action)
            action_visit = node.action_visits[idx]

            # Eq. 8 - back-propagation rule
            #  As states are not explicitly represented in nodes, sometimes termination can occur even though the
            #  current node is non-terminal due to a collision. In this case, we want to force using the reward
            #  to update the Q-values for the occurrence of a collision.
            q = r if child is None or child.is_leaf or force_reward else np.max(child.q_values)
            node.q_values[idx] += (q - node.q_values[idx]) / action_visit
            node.store_q_values()

            key = node.key

    def print(self, node: Node = None):
        """ Print the nodes of the tree recursively starting from the given node. """
        if node is None:
            node = self.root

        logger.debug(f"  {node.key}: (A, Q)={list(zip(node.actions_names, node.q_values))}; Visits={node.action_visits}")
        for child in node.children.values():
            self.print(child)

    def on_finish(self):
        """ Function called when MCTS finishes execution. """
        pass

    @property
    def root(self) -> Node:
        """ Return the root node of the tree. """
        return self._root

    @property
    def tree(self) -> Dict:
        """ The dictionary representing the tree itself. """
        return self._tree

    @property
    def max_depth(self) -> int:
        """ The maximal depth of the search tree. """
        return max([len(k) for k in self._tree])

    @property
    def action_policy(self) -> Policy:
        """ Policy used to select actions during rollouts. Defaults to UCB1. """
        return self._action_policy

    @property
    def plan_policy(self) -> Policy:
        """ Policy used to select the final plan from the tree. Defaults to argmax by Q-values."""
        return self._plan_policy
