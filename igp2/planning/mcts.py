import igp2 as ip
import copy
import traceback
import logging
from typing import List, Dict, Tuple

from igp2.planning.tree import Tree
from igp2.planning.rollout import Rollout
from igp2.planning.node import Node
from igp2.planning.mctsaction import MCTSAction
from igp2.planning.reward import Reward
from igp2.core.util import copy_agents_dict

logger = logging.getLogger(__name__)


class MCTS:
    """ Class implementing single-threaded MCTS search over environment states with macro actions. """

    def __init__(self,
                 scenario_map: ip.Map,
                 n_simulations: int = 30,
                 max_depth: int = 5,
                 reward: Reward = None,
                 open_loop_rollout: bool = False,
                 trajectory_agents: bool = True,
                 fps: int = 10,
                 store_results: str = None,
                 tree_type: type(Tree) = None,
                 node_type: type(Node) = None,
                 action_type: type(MCTSAction) = None,
                 rollout_type: type(Rollout) = None):
        """ Initialise a new MCTS planner over states and macro-actions.

        Args:
            n_simulations: number of rollout simulations to run.
            max_depth: maximum search depth.
            scenario_map: current road layout.
            reward: class to calculate trajectory reward for ego.
            open_loop_rollout: Whether to use open-loop predictions directly instead of closed-loop control.
            trajectory_agents: To use trajectories or plans for non-egos in simulation.
            fps: Rollout simulation frequency.
            tree_type: Type of Tree to use for the search. Allows overwriting standard behaviour.
            node_type: Type of Node to use in the Tree. Allows overwriting standard behaviour.
            rollout_type: Type of Rollout to use for the search. Allows overwriting standard behaviour.
        """
        self.n = n_simulations
        self.d_max = max_depth
        self.scenario_map = scenario_map
        self.reward = reward if reward is not None else Reward()
        self.open_loop_rollout = open_loop_rollout
        self.trajectory_agents = trajectory_agents
        self.fps = fps

        self.tree_type = tree_type if tree_type is not None else Tree
        self.node_type = node_type if node_type is not None else Node
        self.action_type = action_type if action_type is not None else MCTSAction
        self.rollout_type = rollout_type if rollout_type is not None else Rollout

        self.store_results = store_results
        self.results = None
        self.reset_results()

    def reset_results(self):
        """ Resets the stored results in the MCTS instance."""
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
               predictions: Dict[int, ip.GoalsProbabilities],
               debug: bool = False) -> List[MCTSAction]:
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
            for other agents
        """
        self.reset_results()
        self.reward.reset()

        simulator = self.rollout_type(ego_id=agent_id,
                                      initial_frame=frame,
                                      metadata=meta,
                                      scenario_map=self.scenario_map,
                                      fps=self.fps,
                                      open_loop_agents=self.open_loop_rollout,
                                      trajectory_agents=self.trajectory_agents)
        simulator.update_ego_goal(goal)

        # 1. Create tree root from current frame
        root = self.create_node(MCTS.to_key(None), agent_id, frame, goal)
        tree = self.tree_type(root, predictions=predictions)

        for k in range(self.n):
            logger.info(f"MCTS Iteration {k + 1}/{self.n}")
            self._rollout(k, agent_id, goal, tree, simulator, debug, predictions)

            simulator.reset()
            self.reward.reset()

        tree.on_finish()

        final_plan = tree.select_plan()
        logger.info(f"Final plan: {final_plan}")
        tree.print()

        if self.store_results == "final":
            self.results.tree = tree
        elif self.store_results == "all":
            self.results.final_plan = final_plan

        return final_plan

    def _sample_agents(self, aid: int, predictions: Dict[int, ip.GoalsProbabilities]):
        """ Perform sampling of goals and agent trajectories. """
        goal = predictions[aid].sample_goals()[0]
        trajectory, plan = predictions[aid].sample_trajectories_to_goal(goal)
        if trajectory is not None:
            trajectory, plan = trajectory[0], plan[0]
        return goal, trajectory, plan

    def _rollout(self, k: int, agent_id: int, goal: ip.Goal, tree: Tree,
                 simulator: Rollout, debug: bool, predictions: Dict[int, ip.GoalsProbabilities]):
        """ Perform a single rollout of the MCTS search and store results."""
        # 3-6. Sample goal and trajectory
        samples = {}
        for aid, agent in simulator.agents.items():
            if aid == simulator.ego_id:
                continue

            agent_goal, trajectory, plan = self._sample_agents(aid, predictions)
            simulator.update_trajectory(aid, trajectory, plan)
            samples[aid] = (agent_goal, trajectory)
            logger.debug(f"Agent {aid} sample: {plan}")

        tree.set_samples(samples)
        final_key = self._run_simulation(agent_id, goal, tree, simulator, debug)

        if self.store_results == "all":
            logger.debug(f"Storing MCTS search results for iteration {k}.")
            mcts_result = ip.MCTSResult(copy.deepcopy(tree), samples, final_key)
            self.results.add_data(mcts_result)

    def _run_simulation(self, agent_id: int, goal: ip.Goal, tree: Tree, simulator: Rollout, debug: bool) -> tuple:
        depth = 0
        node = tree.root
        key = node.key
        current_frame = node.state
        actions = []

        while depth < self.d_max:
            logger.debug(f"Rollout {depth + 1}/{self.d_max}")
            node.state_visits += 1

            final_frame = None

            # 8. Select applicable macro action with UCB1
            action = tree.select_action(node)
            actions.append(action)
            simulator.update_ego_action(action.macro_action_type, action.ma_args, current_frame)

            logger.debug(f"Action selection: {key} -> {action} from {node.actions_names}")

            # 9. Forward simulate environment
            try:
                trajectory, final_frame, goal_reached, alive, collisions = \
                    simulator.run(current_frame, debug)

                collided_agents_ids = [col.agent_id for col in collisions]
                if self.store_results is not None:
                    agents_copy = copy_agents_dict(simulator.agents, agent_id)
                    node.run_result = ip.RunResult(
                        agents_copy,
                        simulator.ego_id,
                        trajectory,
                        collided_agents_ids,
                        goal_reached,
                        action)

                # 10-16. Reward computation
                r = self.reward(collisions=collisions,
                                alive=alive,
                                ego_trajectory=simulator.agents[agent_id].trajectory_cl if goal_reached else None,
                                goal=goal,
                                depth_reached=depth == self.d_max - 1)
                if r is not None:
                    logger.debug(f"Reward components: {self.reward.reward_components}")

            except Exception as e:
                logger.debug(f"Rollout failed due to error: {str(e)}")
                logger.debug(traceback.format_exc())
                r = -float("inf")

            # Create new node at the end of rollout
            key = MCTS.to_key(actions)

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
        return key

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
        for macro_action in ip.MacroActionFactory.get_applicable_actions(frame[agent_id], self.scenario_map):
            for ma_args in macro_action.get_possible_args(frame[agent_id], self.scenario_map, goal):
                actions.append(self.action_type(macro_action, ma_args))

        node = self.node_type(key, frame, actions)
        node.expand()
        return node

    @staticmethod
    def to_key(plan: List[MCTSAction] = None) -> Tuple[str, ...]:
        """ Convert a list of MCTS actions to an MCTS key. """
        if plan is None:
            return tuple(["Root"])
        return ("Root",) + tuple([action.__repr__() for action in plan])
