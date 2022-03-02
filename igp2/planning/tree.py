import logging
import numpy as np
from typing import Dict, Optional, List, Tuple

from igp2.recognition.goalprobabilities import GoalWithType, GoalsProbabilities
from igp2.results import RewardResult
from igp2.trajectory import VelocityTrajectory
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
                 plan_policy: Policy = None,
                 predictions: Dict[int, GoalsProbabilities] = None):
        """ Initialise a new Tree with the given root.

        Args:
            root: the root node
            action_policy: policy for selecting actions (default: UCB1)
            plan_policy: policy for selecting the final plan (default: Max)
            predictions: optional goal predictions for vehicles
        """
        self._root = root
        self._tree = {root.key: root}

        self._action_policy = action_policy if action_policy is not None else UCB1()
        self._plan_policy = plan_policy if plan_policy is not None else MaxPolicy()

        self._predictions = predictions
        self._samples = None  # Field storing goal prediction sampling for other vehicles

        self._reward_results = []

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

    def add_reward_result(self, reward_results: RewardResult):
        """ Add a new reward outcome to the node if the search has ended here. """
        self._reward_results.append(reward_results)

    def add_child(self, parent: Node, child: Node):
        """ Add a new child to the tree and assign it under an existing parent node. """
        if parent.key in self._tree:
            self._add_node(child)
            self._tree[parent.key].add_child(child)
        else:
            logger.warning(f"Parent {parent.key} not in the tree!")

    def select_action(self, node: Node) -> MCTSAction:
        """ Select one of the actions in the node using the specified policy and update node statistics """
        action, idx = self._action_policy.select(node)
        node.action_visits[idx] += 1
        return action

    def select_plan(self) -> List:
        """ Return the best sequence of actions from the root according to the specified policy. """
        plan = []
        node = self.root
        while node is not None and node.state_visits > 0:
            next_action, action_idx = self._plan_policy.select(node)
            plan.append(next_action)
            node = self[tuple(list(node.key) + [next_action.__repr__()])]
        return plan

    def set_samples(self, samples: Dict[int, Tuple[GoalWithType, VelocityTrajectory]]):
        """ Overwrite the currently stored samples in the tree. """
        self._samples = samples

    def backprop(self, r: float, final_key: Tuple):
        """ Back-propagate the reward through the search branches.

        Args:
            r: reward at end of simulation
            final_key: final node key including final action
        """
        key = final_key
        while key != self._root.key:
            node, action, child = (self[key[:-1]], key[-1], self[key])

            idx = node.actions_names.index(action)
            action_visit = node.action_visits[idx]

            # Eq. 8 - back-propagation rule
            q = r if child is None else np.max(child.q_values)
            node.q_values[idx] += (q - node.q_values[idx]) / action_visit
            node.store_q_values()

            key = node.key

    def print(self, node: Node = None):
        """ Print the nodes of the tree recursively starting from the given node. """
        if node is None:
            node = self.root

        logger.debug(f"{node.key}: (A, Q)={list(zip(node.actions_names, node.q_values))}; Visits={node.action_visits}")
        for child in node.children.values():
            self.print(child)

    @property
    def root(self) -> Node:
        """ Return the root node of the tree. """
        return self._root

    @property
    def tree(self) -> Dict:
        """ The dictionary representing the tree itself. """
        return self._tree

    @property
    def predictions(self) -> Dict[int, GoalsProbabilities]:
        """ Predictions associated with this tree. """
        return self._predictions
