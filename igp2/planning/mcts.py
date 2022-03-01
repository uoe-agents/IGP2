import igp2 as ip
import copy
import traceback
import logging
from typing import List, Dict, Tuple

from igp2.planning.tree import Tree
from igp2.planning.simulator import Simulator
from igp2.planning.node import Node
from igp2.planning.mctsaction import MCTSAction

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
                 scenario_map: ip.Map,
                 n_simulations: int = 30,
                 max_depth: int = 5,
                 cost: ip.Cost = None,
                 rewards: Dict[str, float] = None,
                 open_loop_rollout: bool = False,
                 fps: int = 10,
                 store_results: str = None):
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
        self.cost = cost if cost is not None else ip.Cost()
        self.rewards = rewards if rewards is not None else MCTS.DEFAULT_REWARDS
        self.open_loop_rollout = open_loop_rollout
        self.fps = fps

        self.store_results = store_results
        if self.store_results is None:
            self.results = None
        elif self.store_results == 'final':
            self.results = ip.MCTSResult()
        elif self.store_results == 'all':
            self.results = ip.AllMCTSResult()

    def search(self,
               agent_id: int,
               goal: ip.Goal,
               frame: Dict[int, ip.AgentState],
               meta: Dict[int, ip.AgentMetadata],
               predictions: Dict[int, ip.GoalsProbabilities]) -> List[MCTSAction]:
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
        simulator = ip.Simulator(agent_id, frame, meta, self.scenario_map, self.fps, self.open_loop_rollout)
        simulator.update_ego_goal(goal)

        # 1. Create tree root from current frame
        root = self.create_node(("Root",), agent_id, frame, goal)
        tree = ip.Tree(root)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")

            # 3-6. Sample goal and trajectory
            for aid, agent in simulator.agents.items():
                if aid == simulator.ego_id:
                    continue

                agent_goal = predictions[aid].sample_goals()[0]
                agent_trajectory = predictions[aid].sample_trajectories_to_goal(agent_goal)
                if agent_trajectory is not None: agent_trajectory = agent_trajectory[0]
                simulator.update_trajectory(aid, agent_trajectory)

            self._run_simulation(agent_id, goal, tree, simulator)
            simulator.reset()

            if self.store_results == 'all':
                logger.info(f"Storing MCTS search results for iteration {k}.")
                mcts_result = ip.MCTSResult(copy.deepcopy(tree))
                self.results.add_data(mcts_result)

        final_plan = tree.select_plan()
        logger.info(f"Final plan: {final_plan}")
        tree.print()

        if self.store_results == 'final':
            logger.info(f"Storing MCTS search results.")
            self.results.tree = tree

        return final_plan

    def _run_simulation(self, agent_id: int, goal: ip.Goal, tree: Tree, simulator: Simulator):
        depth = 0
        node = tree.root
        key = node.key
        current_frame = node.state

        while depth < self.d_max:
            logger.debug(f"Rollout {depth + 1}/{self.d_max}")
            node.state_visits += 1
            r = None

            # 8. Select applicable macro action with UCB1
            action = tree.select_action(node)
            simulator.update_ego_action(action.macro_action_type, action.ma_args, current_frame)

            logger.debug(f"Action selection: {key} -> {action} from {node.actions_names}")

            # 9. Forward simulate environment
            try:
                trajectory, final_frame, goal_reached, alive, collisions = simulator.run(current_frame)

                collided_agents_ids = [col.agent_id for col in collisions]
                if self.store_results is not None:
                    run_result = ip.RunResult(copy.copy(simulator.agents), simulator.ego_id, trajectory,
                                              collided_agents_ids, goal_reached)
                    node.add_run_result(run_result)

                # 10-16. Reward computation
                if collisions:
                    r = self.rewards["coll"]
                    logger.debug(f"Ego agent collided with agent(s): {collisions}")
                elif not alive:
                    r = self.rewards["dead"]
                    logger.debug(f"Ego died during rollout!")
                elif goal_reached:
                    total_trajectory = simulator.agents[simulator.ego_id].trajectory_cl
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
            key = tuple(list(key) + [action.__repr__()])
            if r is not None:
                logger.info(f"Rollout finished: r={r}; d={depth + 1}")
                tree.backprop(r, key)
                break

            # 20. Update state variables
            current_frame = final_frame
            if key not in tree:
                child = self.create_node(key, agent_id, current_frame, goal)
                tree.add_child(node, child)
            node = tree[key]
            depth += 1

    def create_node(self, key: Tuple, agent_id: int, frame: Dict[int, ip.AgentState], goal: ip.Goal) -> Node:
        """ Create a new node and expand it. """
        actions = []
        for macro_action in ip.MacroAction.get_applicable_actions(frame[agent_id], self.scenario_map):
            for ma_args in macro_action.get_possible_args(frame[agent_id], self.scenario_map, goal):
                actions.append(MCTSAction(macro_action, ma_args))
        node = ip.Node(key, frame, actions)
        node.expand()
        return node
