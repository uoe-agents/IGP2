import numpy as np
import logging
from typing import List, Dict, Tuple, Hashable

from igp2.agent import AgentState, AgentMetadata
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.planning.simulator import Simulator
from igp2.planning.tree import Tree
from igp2.planning.node import Node
from igp2.recognition.goalprobabilities import GoalsProbabilities


logger = logging.getLogger(__name__)


class MCTS:
    """ Class implementing single-threaded MCTS search over environment states with macro actions. """
    DEFAULT_REWARDS = {
        "coll": -1,
        "term": -1
    }

    def __init__(self,
                 scenario_map: Map,
                 n_simulations: int = 30,
                 max_depth: int = 5,
                 rewards: Dict[str, float] = None):
        """ Initialise a new MCTS planner over states and macro-actions.

        Args:
            n_simulations: number of rollout simulations to run
            max_depth: maximum search depth
            scenario_map: current road layout
            rewards: dictionary giving the reward values for simulation outcomes
        """
        self.n = n_simulations
        self.d_max = max_depth
        self.scenario_map = scenario_map
        self.rewards = rewards if rewards is not None else MCTS.DEFAULT_REWARDS

    def search(self,
               agent_id: int,
               frame: Dict[int, AgentState],
               meta: Dict[int, AgentMetadata],
               predictions: Dict[int, GoalsProbabilities]) -> List[MacroAction]:
        """ Run MCTS search for the given agent

        Args:
            agent_id: agent to plan for
            frame: current (observed) state of the environment
            meta: metadata of agents present in frame
            predictions: dictionary of goal predictions for agents in frame

        Returns:
            a list of macro actions encoding the optimal plan for the ego agent given the current goal predictions
            for other agents
        """
        simulator = Simulator(agent_id, frame, meta, self.scenario_map)

        # 1. Create tree root from current frame
        root = self.create_node(("Root",), agent_id, frame)
        tree = Tree(root)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")

            # 3-6. Sample goal and trajectory
            for aid, agent in simulator.agents.items():
                if aid == simulator.ego_id:
                    continue

                goal = predictions[aid].sample_goals()[0]
                trajectory = predictions[aid].sample_trajectories_to_goal(goal)[0]
                simulator.update_trajectory(aid, trajectory)

            self._run_simulation(agent_id, tree, simulator)

    def _run_simulation(self, agent_id: int, tree: Tree, simulator: Simulator):
        depth = 0
        node = tree.root
        key = node.key

        while depth < self.d_max:
            # 8. Select applicable macro action with UCB1
            macro_action = tree.select_action(node)

            # 9. Forward simulate environment
            simulator.update_ego_action(macro_action)
            new_frame, done, collision_id = simulator.run()

            # 10-16. Reward computation
            r = None
            if collision_id is not None:
                r = self.rewards["coll"]
            elif done:
                r = 0.0  # TODO
            elif depth == self.d_max - 1:
                r = self.rewards["term"]

            # 17-19. Backpropagation
            if r is not None:
                tree.backprop(key)
                break

            # 20. Update state variables
            key = tuple(list(key) + [macro_action.__name__])
            if key not in tree:
                child = self.create_node(key, agent_id, new_frame)
                tree.add_child(node, child)
            node = tree[key]
            depth += 1

        return

    def create_node(self, key: Hashable, agent_id: int, frame: Dict[int, AgentState]) -> Node:
        """ Create a new node and expand it. """
        actions = MacroAction.get_applicable_actions(frame[agent_id], self.scenario_map)
        node = Node(key, frame, actions)
        node.expand()
        return node
