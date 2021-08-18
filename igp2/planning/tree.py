import logging
from typing import Dict, Optional

from igp2.planning.node import Node
from igp2.planning.policy import Policy, UCB1

logger = logging.getLogger(__name__)


class Tree:
    """ Defines an abstract tree consisting of nodes.

    A Tree is represented as a dictionary where each key corresponds to a key of the node and the values are
    the corresponding node object. A node is a leaf if it has no children.
    """

    def __init__(self, root: Node, action_policy: Policy = None):
        """ Initialise a new Tree with the given root"""
        self._root = root
        self._tree = {root.key: root}

        self._action_policy = action_policy if action_policy is not None else UCB1()

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
        """ Select one of the actions in the node using the specified policy """
        return self._action_policy.select(node)

    @property
    def root(self) -> Node:
        """ Return the root node of the tree"""
        return self._root

    @property
    def tree(self) -> Dict:
        return self._tree
