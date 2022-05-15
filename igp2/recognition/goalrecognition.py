import numpy as np

import igp2 as ip
import logging
from typing import Dict, List, Tuple

from igp2.recognition.astar import AStar
from igp2.recognition.goalprobabilities import GoalsProbabilities

logger = logging.getLogger(__name__)


class GoalRecognition:
    """This class updates existing goal probabilities using likelihoods computed from the vehicle current trajectory.
    It also calculates the probabilities of up to n_trajectories paths to these goals. """

    def __init__(self, astar: AStar, smoother: ip.VelocitySmoother, scenario_map: ip.Map, cost: ip.Cost = None,
                 n_trajectories: int = 1, beta: float = 1., gamma=1, reward_as_difference: bool = True):
        """Initialises a goal recognition class that will be used to update a GoalProbabilities object.
        
        Args:
            astar: AStar object used to generate trajectories
            smoother: Velocity smoother object used to make the AStar generated trajectories realistic
            scenario_map: a Map object representing the current scenario
            cost: a Cost object representing how the reward associated to each trajectory will be computed.
            beta: scaling parameter for the Boltzmann distribution generating the likelihoods
            gamma: scaling parameter for the Boltzmann distribution generating trajectory probabilities
            reward_as_difference: choose if we define the reward for each trajectory separately or if the
                                  reward is computed from differences of the different trajectory 
                                  quantities alongside the pathlength.
            n_trajectories: The number of trajectories to try to return
        """
        self._n_trajectories = n_trajectories
        self._beta = beta
        self._gamma = gamma
        self._reward_as_difference = reward_as_difference
        self._astar = astar
        self._smoother = smoother
        self._cost = ip.Cost() if cost is None else cost
        self._scenario_map = scenario_map

    def update_goals_probabilities(self,
                                   goals_probabilities: GoalsProbabilities,
                                   observed_trajectory: ip.Trajectory,
                                   agent_id: int,
                                   frame_ini: Dict[int, ip.AgentState],
                                   frame: Dict[int, ip.AgentState],
                                   maneuver: ip.Maneuver = None,
                                   visible_region: ip.Circle = None) -> GoalsProbabilities:
        """Updates the goal probabilities, and stores relevant information in the GoalsProbabilities object.
        
        Args: 
            goals_probabilities: GoalsProbabilities object to update
            observed_trajectory: current vehicle trajectory
            agent_id: id of agent in current frame
            frame_ini: frame corresponding to the first state of the agent's trajectory
            frame: current frame
            maneuver: current maneuver in execution by the agent
            visible_region: region of the map which is visible to the ego vehicle
        """
        norm_factor = 0.
        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            try:
                goal = goal_and_type[0]

                if goal.reached(frame_ini[agent_id].position):
                    raise RuntimeError(f"Agent {agent_id} reached goal at start.")

                # 4. and 5. Generate optimum trajectory from initial point and smooth it
                if goals_probabilities.optimum_trajectory[goal_and_type] is None:
                    logger.debug("Generating optimum trajectory")
                    trajectories, plans = self._generate_trajectory(1, agent_id, frame_ini, goal,
                                                                    state_trajectory=None,
                                                                    visible_region=visible_region)
                    goals_probabilities.optimum_trajectory[goal_and_type] = trajectories[0]
                    goals_probabilities.optimum_plan[goal_and_type] = plans[0]

                opt_trajectory = goals_probabilities.optimum_trajectory[goal_and_type]

                # 7. and 8. Generate optimum trajectory from last observed point and smooth it
                all_trajectories, all_plans = self._generate_trajectory(
                    self._n_trajectories, agent_id, frame, goal, observed_trajectory,
                    visible_region=visible_region)

                # 6. Calculate optimum reward
                goals_probabilities.optimum_reward[goal_and_type] = self._reward(opt_trajectory, goal)

                # For each generated possible trajectory to this goal
                for trajectory in all_trajectories:
                    # join the observed and generated trajectories
                    trajectory.insert(observed_trajectory)

                    # 9,10. calculate rewards, likelihood
                    reward = self._reward(trajectory, goal)
                    goals_probabilities.all_rewards[goal_and_type].append(reward)

                    reward_diff = self._reward_difference(opt_trajectory, trajectory, goal)
                    goals_probabilities.all_reward_differences[goal_and_type].append(reward_diff)

                # 11. Calculate likelihood
                likelihood = self._likelihood(opt_trajectory, all_trajectories[0], goal)

                # Calculate all trajectory probabilities
                goals_probabilities.trajectories_probabilities[goal_and_type] = \
                    self._trajectory_probabilities(goals_probabilities.all_rewards[goal_and_type])

                # Write additional goals probabilities fields
                goals_probabilities.all_trajectories[goal_and_type] = all_trajectories
                goals_probabilities.all_plans[goal_and_type] = all_plans
                goals_probabilities.current_trajectory[goal_and_type] = all_trajectories[0]
                goals_probabilities.reward_difference[goal_and_type] = \
                    goals_probabilities.all_reward_differences[goal_and_type][0]
                goals_probabilities.current_reward[goal_and_type] = \
                    goals_probabilities.all_rewards[goal_and_type][0]

            except RuntimeError as e:
                logger.debug(str(e))
                likelihood = 0.
                goals_probabilities.current_trajectory[goal_and_type] = None

            # update goal probabilities
            goals_probabilities.goals_probabilities[goal_and_type] = goals_probabilities.goals_priors[
                                                                         goal_and_type] * likelihood
            goals_probabilities.likelihood[goal_and_type] = likelihood
            norm_factor += likelihood * goals_probabilities.goals_priors[goal_and_type]

        # then divide prob by norm_factor to normalise
        for key, prob in goals_probabilities.goals_probabilities.items():
            try:
                goals_probabilities.goals_probabilities[key] = prob / norm_factor
            except ZeroDivisionError as e:
                logger.debug("All goals unreachable. Setting all probabilities to 0.")
                break

        return goals_probabilities

    def _generate_trajectory(self,
                             n_trajectories: int,
                             agent_id: int,
                             frame: Dict[int, ip.AgentState],
                             goal: ip.Goal,
                             state_trajectory: ip.Trajectory,
                             visible_region: ip.Circle = None) \
            -> Tuple[List[ip.VelocityTrajectory], List[List[ip.MacroAction]]]:
        """Generates up to n possible trajectories from the current frame of an agent to the specified goal"""
        trajectories, plans = self._astar.search(agent_id, frame, goal,
                                                 self._scenario_map,
                                                 n_trajectories,
                                                 open_loop=True,
                                                 visible_region=visible_region)
        if len(trajectories) == 0:
            raise RuntimeError(f"{goal} is unreachable")

        for trajectory in trajectories:
            if state_trajectory is None:
                trajectory.velocity[0] = frame[agent_id].speed  # Optimal case
            else:
                trajectory.velocity[0] = state_trajectory.velocity[-1]
            self._smoother.load_trajectory(trajectory)
            trajectory.velocity = self._smoother.split_smooth()

        return trajectories, plans

    def _trajectory_probabilities(self, rewards: List[float]) -> List[float]:
        """ Calculate the probabilities of each plausible trajectory given their rewards """
        rewards = np.array(rewards)
        num = np.exp(self._gamma * rewards - np.max(rewards))
        return list(num / np.sum(num))

    def _likelihood(self, optimum_trajectory: ip.Trajectory, current_trajectory: ip.Trajectory, goal: ip.Goal) -> float:
        """Calculates the non normalised likelihood for a specified goal"""
        difference = self._reward_difference(optimum_trajectory, current_trajectory, goal)
        return float(np.clip(np.exp(self._beta * difference), 1e-305, 1e305))

    def _reward(self, trajectory: ip.Trajectory, goal: ip.Goal) -> float:
        """Calculates the reward associated to a trajectory for a specified goal."""
        return -self._cost.trajectory_cost(trajectory, goal)

    def _reward_difference(self, optimum_trajectory: ip.Trajectory, current_trajectory: ip.Trajectory, goal: ip.Goal):
        """If reward_as_difference is True, calculates the reward as a measure of similarity between the two 
        trajectories' attributes. Otherwise simply calculates the difference as the difference of the 
        individual rewards"""
        if self._reward_as_difference:
            return -self._cost.cost_difference_resampled(optimum_trajectory, current_trajectory, goal)
        else:
            return self._reward(current_trajectory, goal) - self._reward(optimum_trajectory, goal)
