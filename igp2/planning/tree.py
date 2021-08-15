import logging

from igp2.planning.node import Node

logger = logging.getLogger(__name__)


class Tree:
    """ Defines an abstract tree consisting of nodes.

    A Tree represented as a dictionary where each key corresponds to a node in the tree and the values of the key
    define the children of that node. A node is a leaf if it has no children.
    """

    def __init__(self, root: Node):
        """ Initialise a new Tree with the given root"""
        self._root = root
        self._tree = {}

    def add_node(self, node: Node):
        """ Add a new node to the tree if not already in the tree. """
        if node.state not in self._tree:
            self._tree[node.state] = node
        else:
            logger.warning(f"Node {node.state} already in the tree!")

    def add_child(self, parent: Node, child: Node):
        """ Add a new child to an existing parent node. """
        if parent.state in self._tree:
            self._tree[parent.state].add_child(child)
        else:
            logger.warning(f"Parent {parent.state} not in the tree!")

    def __contains__(self, item: Node) -> bool:
        return item in self._tree
