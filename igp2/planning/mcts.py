import igp2 as ip
import copy
import traceback
import logging
from typing import List, Dict, Tuple

from igp2.planning.tree import Tree
from igp2.planning.simulator import Simulator
from igp2.planning.node import Node
from igp2.planning.mctsaction import MCTSAction
from igp2.planning.reward import Reward

logger = logging.getLogger(__name__)


def copy_agents_dict(agents_dict, agent_id):
    # Remove temporarily due to circular dependency
    current_ma_tmp = agents_dict[agent_id].current_macro
    agents_dict[agent_id]._current_macro = None
    memo = {}
    for aid, agent in agents_dict.items():
        if aid == agent_id:
            continue
        memo[aid] = agent._maneuver
        agent._maneuver = None

    agents_copy = copy.deepcopy(agents_dict)

    agents_dict[agent_id]._current_macro = current_ma_tmp
    agents_copy[agent_id]._current_macro = current_ma_tmp
    for aid, agent in agents_dict.items():
        if aid == agent_id:
            continue
        agent._maneuver = memo[aid]
        agents_copy[aid]._maneuver = memo[aid]
    return agents_copy


class MCTS:
    """ Class implementing single-threaded MCTS search over environment states with macro actions. """

    def __init__(self,
                 scenario_map: ip.Map,
                 n_simulations: int = 30,
                 max_depth: int = 5,
                 reward: Reward = None,
                 open_loop_rollout: bool = False,
                 fps: int = 10,
                 store_results: str = None,
                 tree_type: type(Tree) = None,
                 node_type: type(Node) = None):
        """ Initialise a new MCTS planner over states and macro-actions.

        Args:
            n_simulations: number of rollout simulations to run
            max_depth: maximum search depth
            scenario_map: current road layout
            reward: class to calculate trajectory reward for ego
            open_loop_rollout: Whether to use open-loop predictions directly instead of closed-loop control
            fps: Rollout simulation frequency
            tree_type: Type of Tree to use for the search. Allows overwriting standard behaviour.
            node_type: Type of Node to use in the Tree. Allows overwriting standard behaviour.
        """
        self.n = n_simulations
        self.d_max = max_depth
        self.scenario_map = scenario_map
        self.reward = reward if reward is not None else Reward()
        self.open_loop_rollout = open_loop_rollout
        self.fps = fps

        self.tree_type = tree_type if tree_type is not None else Tree
        self.node_type = node_type if node_type is not None else Node

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
        simulator = Simulator(agent_id, frame, meta, self.scenario_map, self.fps, self.open_loop_rollout)
        simulator.update_ego_goal(goal)

        # 1. Create tree root from current frame
        root = self.create_node(("Root",), agent_id, frame, goal)
        tree = self.tree_type(root, predictions=predictions)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")

            # 3-6. Sample goal and trajectory
            samples = {}
            for aid, agent in simulator.agents.items():
                if aid == simulator.ego_id:
                    continue

                agent_goal = predictions[aid].sample_goals()[0]
                agent_trajectory = predictions[aid].sample_trajectories_to_goal(agent_goal)
                if agent_trajectory is not None: agent_trajectory = agent_trajectory[0]
                simulator.update_trajectory(aid, agent_trajectory)
                samples[aid] = (agent_goal, agent_trajectory)

            tree.set_samples(samples)
            self._run_simulation(agent_id, goal, tree, simulator)

            if self.store_results == 'all':
                logger.info(f"Storing MCTS search results for iteration {k}.")
                mcts_result = ip.MCTSResult(copy.deepcopy(tree), samples)
                self.results.add_data(mcts_result)

            simulator.reset()
            self.reward.reset()

        tree.on_finish()

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

            reward_result = None
            final_frame = None

            # 8. Select applicable macro action with UCB1
            action = tree.select_action(node)
            simulator.update_ego_action(action.macro_action_type, action.ma_args, current_frame)

            logger.debug(f"Action selection: {key} -> {action} from {node.actions_names}")

            # 9. Forward simulate environment
            try:
                trajectory, final_frame, goal_reached, alive, collisions = simulator.run(current_frame)

                collided_agents_ids = [col.agent_id for col in collisions]
                if self.store_results is not None:
                    agents_copy = copy_agents_dict(simulator.agents, agent_id)
                    run_result = ip.RunResult(
                        agents_copy,
                        simulator.ego_id,
                        trajectory,
                        collided_agents_ids,
                        goal_reached)
                    node.add_run_result(run_result)

                # 10-16. Reward computation
                r = self.reward(collisions=collisions,
                                alive=alive,
                                ego_trajectory=simulator.agents[agent_id].trajectory_cl if goal_reached else None,
                                goal=goal,
                                depth_reached=depth == self.d_max - 1)

            except Exception as e:
                logger.debug(f"Rollout failed due to error: {str(e)}")
                logger.debug(traceback.format_exc())
                r = -float("inf")

            # Create new node at the end of rollout
            key = tuple(list(key) + [action.__repr__()])

            # 17-19. Back-propagation
            if r is not None:
                logger.info(f"Rollout finished: r={r}; d={depth + 1}")
                node.add_reward_result(key, copy.deepcopy(self.reward))
                tree.backprop(r, key)
                break

            # 20. Update state variables
            current_frame = final_frame
            if key not in tree:
                child = self.create_node(key, agent_id, current_frame, goal)
                tree.add_child(node, child)
            node = tree[key]
            depth += 1

    def create_node(self,
                    key: Tuple,
                    agent_id: int,
                    frame: Dict[int, ip.AgentState],
                    goal: ip.Goal) -> Node:
        """ Create a new node and expand it.

        Args:
            key: Key to assign to the node
            agent_id: Agent we are searching for
            frame: Current state of the environment
            goal: Goal of the agent with agent_id
        """
        actions = []
        for macro_action in ip.MacroAction.get_applicable_actions(frame[agent_id], self.scenario_map):
            for ma_args in macro_action.get_possible_args(frame[agent_id], self.scenario_map, goal):
                actions.append(MCTSAction(macro_action, ma_args))

        node = self.node_type(key, frame, actions)
        node.expand()
        return node
