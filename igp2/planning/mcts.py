import traceback

import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Hashable

from igp2.agentstate import AgentState, AgentMetadata
from igp2.cost import Cost
from igp2.goal import Goal
from igp2.opendrive.map import Map
from igp2.planlibrary.macro_action import MacroAction
from igp2.planning.simulator import Simulator
from igp2.planning.tree import Tree
from igp2.planning.node import Node
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.trajectory import StateTrajectory

logger = logging.getLogger(__name__)


class MCTS:
    """ Class implementing single-threaded MCTS search over environment states with macro actions. """
    DEFAULT_REWARDS = {
        "coll": -100,
        "term": -100,
        "dead": -100,
        "err": -1000
    }

    def __init__(self,
                 scenario_map: Map,
                 n_simulations: int = 30,
                 max_depth: int = 5,
                 cost: Cost = None,
                 rewards: Dict[str, float] = None,
                 open_loop_rollout: bool = False,
                 fps: int = 10):
        """ Initialise a new MCTS planner over states and macro-actions.

        Args:
            n_simulations: number of rollout simulations to run
            max_depth: maximum search depth
            scenario_map: current road layout
            cost: class to calculate trajectory cost for ego
            rewards: dictionary giving the reward values for simulation outcomes
            open_loop_rollout: Whether to use open-loop predictions directly instead of closed-loop control
            fps: Rollout simulation frequency
        """
        self.n = n_simulations
        self.d_max = max_depth
        self.scenario_map = scenario_map
        self.cost = cost if cost is not None else Cost()
        self.rewards = rewards if rewards is not None else MCTS.DEFAULT_REWARDS
        self.open_loop_rollout = open_loop_rollout
        self.fps = fps

    def search(self,
               agent_id: int,
               goal: Goal,
               frame: Dict[int, AgentState],
               meta: Dict[int, AgentMetadata],
               predictions: Dict[int, GoalsProbabilities]) -> List[MacroAction]:
        """ Run MCTS search for the given agent

        Args:
            agent_id: agent to plan for
            goal: end goal of the vehicle
            frame: current (observed) state of the environment
            meta: metadata of agents present in frame
            predictions: dictionary of goal predictions for agents in frame

        Returns:
            a list of macro actions encoding the optimal plan for the ego agent given the current goal predictions
            for other agents
        """
        simulator = Simulator(agent_id, frame, meta, self.scenario_map, self.fps, self.open_loop_rollout)
        simulator.update_ego_goal(goal)

        # 1. Create tree root from current frame
        root = self.create_node(("Root",), agent_id, frame)
        tree = Tree(root)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")

            # 3-6. Sample goal and trajectory
            for aid, agent in simulator.agents.items():
                if aid == simulator.ego_id:
                    continue

                agent_goal = predictions[aid].sample_goals()[0]
                agent_trajectory = predictions[aid].sample_trajectories_to_goal(agent_goal)[0]
                simulator.update_trajectory(aid, agent_trajectory)

            self._run_simulation(agent_id, goal, tree, simulator)
            simulator.reset()

        final_plan = tree.select_plan()
        logger.info(f"Final plan: {final_plan}")
        tree.print()

        return final_plan

    def _run_simulation(self, agent_id: int, goal: Goal, tree: Tree, simulator: Simulator):
        depth = 0
        node = tree.root
        key = node.key
        current_frame = node.state
        total_trajectory = StateTrajectory(simulator.fps)

        while depth < self.d_max:
            logger.debug(f"Rollout {depth + 1}/{self.d_max}")
            node.state_visits += 1
            r = None

            # 8. Select applicable macro action with UCB1
            macro_action = tree.select_action(node)

            logger.debug(f"Action selection: {key} -> {macro_action.__name__} from {node.actions_names}")

            # 9. Forward simulate environment
            try:
                simulator.update_ego_action(macro_action, current_frame)
                trajectory, final_frame, goal_reached, alive, collisions = simulator.run(current_frame)
                total_trajectory.extend(trajectory, reload_path=False)

                # 10-16. Reward computation
                if collisions:
                    r = self.rewards["coll"]
                    logger.debug(f"Ego agent collided with agent(s): {collisions}")
                elif not alive:
                    r = self.rewards["dead"]
                    logger.debug(f"Ego died during rollout!")
                elif goal_reached:
                    total_trajectory.calculate_path_and_velocity()
                    r = -self.cost.trajectory_cost(total_trajectory, goal)
                    logger.debug(f"Goal {goal} reached!")
                elif depth == self.d_max - 1:
                    r = self.rewards["term"]
                    logger.debug("Reached final rollout depth!")

            except Exception as e:
                logger.debug(f"Rollout failed due to error: {str(e)}")
                logger.debug(traceback.format_exc())
                r = self.rewards["err"]

            # 17-19. Back-propagation
            key = tuple(list(key) + [macro_action.__name__])
            if r is not None:
                logger.info(f"Rollout finished: r={r}; d={depth + 1}")
                tree.backprop(r, key)
                break

            # 20. Update state variables
            if key not in tree:
                child = self.create_node(key, agent_id, final_frame)
                tree.add_child(node, child)
            current_frame = final_frame
            node = tree[key]
            depth += 1

    def create_node(self, key: Tuple, agent_id: int, frame: Dict[int, AgentState]) -> Node:
        """ Create a new node and expand it. """
        actions = MacroAction.get_applicable_actions(frame[agent_id], self.scenario_map)
        node = Node(key, frame, actions)
        node.expand()
        return node
