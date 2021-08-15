import numpy as np
import logging
from typing import List, Dict, Tuple

from igp2.agent import AgentState
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.planning.simulator import Simulator
from igp2.planning.tree import Tree
from igp2.planning.node import Node
from igp2.recognition.goalprobabilities import GoalsProbabilities


logger = logging.getLogger(__name__)

class MCTS:
    """ Class implementing single-threaded MCTS search over environment states with macro actions. """

    def __init__(self, scenario_map: Map, n_simulations: int = 30, max_depth: int = 5):
        """ Initialise a new MCTS planner over states and macro-actions.

        Args:
            n_simulations: number of rollout simulations to run
            max_depth: maximum search depth
            scenario_map: current road layout
        """
        self.n = n_simulations
        self.d_max = max_depth
        self.scenario_map = scenario_map

    def search(self,
               agent_id: int,
               frame: Dict[int, AgentState],
               predictions: GoalsProbabilities) -> List[MacroAction]:
        """ Run MCTS search for the given agent

        Args:
            agent_id: agent to plan for
            frame: current (observed) state of the environment
            predictions: goal predictions for agents in frame

        Returns:
            a list of macro actions encoding the optimal plan for the ego agent given the current goal predictions
            for other agents
        """
        simulator = Simulator(agent_id, frame, self.scenario_map)

        root = self.create_node(agent_id, frame)
        tree = Tree(root)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")


    def _simulate(self, tree: Tree, simulator: Simulator):
        depth = 0

        return

    def create_node(self, agent_id: int, frame: Dict[int, AgentState]) -> Node:
        """ Create a new node and expand it. """
        state = tuple([state.to_hashable() for agent_id, state in frame.items()])
        actions = MacroAction.get_applicable_actions(frame[agent_id], self.scenario_map)
        node = Node(state, actions)
        node.expand()
        return node
