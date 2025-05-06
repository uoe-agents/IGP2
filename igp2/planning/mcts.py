import copy
import traceback
import logging
from typing import List, Dict, Tuple

from igp2.opendrive.map import Map
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.planning.tree import Tree
from igp2.planning.rollout import Rollout
from igp2.planning.node import Node
from igp2.planning.mctsaction import MCTSAction
from igp2.planning.reward import Reward
from igp2.planlibrary.macro_action import MacroActionFactory
from igp2.core.results import MCTSResult, AllMCTSResult, RunResult
from igp2.core.util import copy_agents_dict
from igp2.core.goal import Goal
from igp2.core.agentstate import AgentState, AgentMetadata

logger = logging.getLogger(__name__)


class MCTS:
    """ Class implementing single-threaded MCTS search over environment states with macro actions. """

    def __init__(self,
                 scenario_map: Map,
                 n_simulations: int = 30,
                 max_depth: int = 5,
                 reward: Reward = None,
                 open_loop_rollout: bool = False,
                 trajectory_agents: bool = True,
                 fps: int = 10,
                 env_fps: int = 20,
                 store_results: str = None,
                 **kwargs):
        """ Initialise a new MCTS planner over states and macro-actions.

        Args:
            n_simulations: number of rollout simulations to run.
            max_depth: maximum search depth.
            scenario_map: current road layout.
            reward: class to calculate trajectory reward for ego.
            open_loop_rollout: Whether to use open-loop predictions directly instead of closed-loop control.
            trajectory_agents: To use trajectories or plans for non-egos in simulation.
            fps: Rollout simulation frequency.
            env_fps: Environment simulation frequency.

        Keyword Args:
            tree_type: Type of Tree to use for the search. Allows overwriting standard behaviour.
            node_type: Type of Node to use in the Tree. Allows overwriting standard behaviour.
            action_type: Type of MCTSAction to use for the search. Allows overwriting standard behaviour.
            rollout_type: Type of Rollout simulator to use for the search. Allows overwriting standard behaviour.
        """
        self.n = n_simulations
        self.d_max = max_depth
        self.scenario_map = scenario_map
        self.reward = reward if reward is not None else Reward()
        self.open_loop_rollout = open_loop_rollout
        self.trajectory_agents = trajectory_agents
        self.fps = fps
        self.env_fps = env_fps

        self.tree_type = kwargs.get("tree_type", Tree)
        self.node_type = kwargs.get("node_type", Node)
        self.action_type = kwargs.get("action_type", MCTSAction)
        self.rollout_type = kwargs.get("rollout_type", Rollout)

        self.store_results = store_results
        self.results = None

        self.reset()

    def search(self,
               agent_id: int,
               goal: Goal,
               frame: Dict[int, AgentState],
               meta: Dict[int, AgentMetadata],
               predictions: Dict[int, GoalsProbabilities],
               debug: bool = False) -> Tuple[List[MCTSAction], Tree]:
        """ Run MCTS search for the given agent

        Args:
            agent_id: agent to plan for
            goal: end goal of the vehicle
            frame: current (observed) state of the environment
            meta: metadata of agents present in frame
            predictions: dictionary of goal predictions for agents in frame
            debug: Whether to plot rollouts.

        Returns:
            a list of macro actions encoding the optimal plan for the ego agent given the current goal predictions
            for other agents and the search tree.
        """
        self.reset()

        simulator = self.rollout_type(ego_id=agent_id,
                                      initial_frame=frame,
                                      metadata=meta,
                                      scenario_map=self.scenario_map,
                                      fps=self.fps,
                                      open_loop_agents=self.open_loop_rollout,
                                      trajectory_agents=self.trajectory_agents)
        simulator.update_ego_goal(goal)

        # 1. Create tree root from current frame
        tree = self._create_tree(agent_id, frame, goal, predictions)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")
            self._rollout(k, agent_id, goal, tree, simulator, debug, predictions)

            simulator.reset()
            self.reward.reset()

        tree.on_finish()

        final_plan, optimal_trace = tree.select_plan()
        logger.info(f"Final plan: {final_plan}")
        tree.print()

        if self.store_results == "final":
            self.results.tree = tree
        elif self.store_results == "all":
            self.results.final_plan = final_plan
            self.results.predictions = predictions
            self.results.optimal_trace = optimal_trace

        return final_plan, tree

    def _sample_agents(self, aid: int, predictions: Dict[int, GoalsProbabilities]):
        """ Perform sampling of goals and agent trajectories. """
        goal = predictions[aid].sample_goals()[0]
        trajectory, plan = predictions[aid].sample_trajectories_to_goal(goal)
        if trajectory is not None:
            trajectory, plan = trajectory[0], plan[0]
        return goal, trajectory, plan

    def _reset_results(self):
        """ Resets the stored results in the MCTS instance."""
        if self.store_results is None:
            self.results = None
        elif self.store_results == 'final':
            self.results = MCTSResult()
        elif self.store_results == 'all':
            self.results = AllMCTSResult()

    def reset(self):
        """ Reset the MCTS planner. """
        self._reset_results()
        self.reward.reset()

    def _create_tree(self,
                     agent_id: int,
                     frame: Dict[int, AgentState],
                     goal: Goal,
                     predictions: Dict[int, GoalsProbabilities]):
        """ Creates a new MCTS tree to store results. """
        root = self._create_node(self.to_key(None), agent_id, frame, goal)
        tree = self.tree_type(root)
        return tree

    def _rollout(self, k: int, agent_id: int, goal: Goal, tree: Tree,
                 simulator: Rollout, debug: bool, predictions: Dict[int, GoalsProbabilities]):
        """ Perform a single rollout of the MCTS search and store results."""
        # 3-6. Sample goal and trajectory
        samples = {}
        failed = []
        for aid, agent in simulator.agents.items():
            if aid == simulator.ego_id:
                continue

            try:
                agent_goal, trajectory, plan = self._sample_agents(aid, predictions)
                simulator.update_trajectory(aid, trajectory, plan)
                samples[aid] = (agent_goal, trajectory)
                logger.debug(f" Agent {aid} sample: {plan}")
            except ValueError as e:
                logger.debug(f"  Agent {aid} failed to sample goal: {str(e)}")
                failed.append(aid)

        for failed_id in failed:
            del simulator.agents[failed_id]

        final_key = self._run_simulation(agent_id, goal, tree, simulator, debug)
        logger.debug(f"  Final key: {final_key}")

        if self.store_results == "all":
            logger.debug(f"  Storing MCTS search results for iteration {k}.")
            mcts_result = MCTSResult(copy.deepcopy(tree), samples, final_key)
            self.results.add_data(mcts_result)

    def _run_simulation(self, agent_id: int, goal: Goal, tree: Tree, simulator: Rollout, debug: bool) -> tuple:
        depth = 0
        node = tree.root
        key = node.key
        current_frame = node.state
        actions = []

        while depth < self.d_max:
            logger.debug(f"    Rollout {depth + 1}/{self.d_max}")
            node.state_visits += 1

            final_frame = None
            force_reward = False

            try:
                # 8. Select applicable macro action with UCB1
                action = tree.select_action(node)
                actions.append(action)
                simulator.update_ego_action(action.macro_action_type, action.ma_args, current_frame)

                logger.debug(f"    Action selection: {action} from {node.actions_names} in {key}")

                # 9. Forward simulate environment
                trajectory, final_frame, goal_reached, alive, collisions = \
                    simulator.run(current_frame, debug)

                collided_agents_ids = [col.agent_id for col in collisions]
                if self.store_results is not None:
                    agents_copy = copy_agents_dict(simulator.agents, agent_id)
                    node.run_result = RunResult(
                        agents_copy,
                        simulator.ego_id,
                        trajectory,
                        collided_agents_ids,
                        goal_reached,
                        action)

                # 10-16. Reward computation
                ego_trajectory = simulator.agents[agent_id].trajectory_cl if goal_reached else None
                depth_reached = depth == self.d_max - 1
                r = self.reward(collisions=collisions,
                                alive=alive,
                                ego_trajectory=ego_trajectory,
                                goal=goal,
                                depth_reached=depth_reached)
                if collisions:
                    logger.debug(f"    Ego agent collided with agent(s): {collisions}")
                elif not alive:
                    logger.debug(f"    Ego died during rollout!")
                elif ego_trajectory is not None and goal is not None:
                    logger.debug(f"    Goal reached!")
                elif depth_reached:
                    logger.debug("    Reached final rollout depth!")
                if r is not None:
                    logger.debug(f"    Reward components: {self.reward.reward_components}")
                    force_reward = len(collisions) > 0

            except Exception as e:
                logger.debug(f"    Rollout failed due to error: {str(e)}")
                logger.debug(traceback.format_exc())
                r = -float("inf")

            # Create new node at the end of rollout
            key = self.to_key(actions)

            # 17-19. Back-propagation
            if r is not None:
                logger.info(f"    Rollout finished: r={r}; d={depth + 1}")
                node.add_reward_result(key, copy.deepcopy(self.reward))
                tree.backprop(r, key, force_reward)
                break

            # 20. Update state variables
            current_frame = final_frame
            if key not in tree:
                child = self._create_node(key, agent_id, current_frame, goal)
                tree.add_child(node, child)
            node = tree[key]
            depth += 1
        return key

    def _create_node(self,
                     key: Tuple,
                     agent_id: int,
                     frame: Dict[int, AgentState],
                     goal: Goal) -> Node:
        """ Create a new node and expand it.

        Args:
            key: Key to assign to the node
            agent_id: Agent we are searching for
            frame: Current state of the environment
            goal: Goal of the agent with agent_id
        """
        actions = []
        for macro_action in MacroActionFactory.get_applicable_actions(frame[agent_id], self.scenario_map):
            for ma_args in macro_action.get_possible_args(frame[agent_id], self.scenario_map, goal):
                actions.append(self.action_type(macro_action, ma_args))

        node = self.node_type(key, frame, actions[::-1])
        node.expand()
        return node

    def to_key(self, plan: List[MCTSAction] = None) -> Tuple[str, ...]:
        """ Convert a list of MCTS actions to an MCTS key. """
        if plan is None:
            return tuple(["Root"])
        return ("Root",) + tuple([str(action) for action in plan])
