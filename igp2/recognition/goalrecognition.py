import copy

import numpy as np

import igp2 as ip
import logging
from typing import Dict, List, Tuple

from igp2.recognition.astar import AStar
from igp2.recognition.goalprobabilities import GoalsProbabilities
from shapely.geometry import Point

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
                                   visible_region: ip.Circle = None,
                                   search_nearby_lanes = True) -> GoalsProbabilities:
        """Updates the goal probabilities, and stores relevant information in the GoalsProbabilities object.
        
        Args: 
            goals_probabilities: GoalsProbabilities object to update
            observed_trajectory: current vehicle trajectory
            agent_id: id of agent in current frame
            frame_ini: frame corresponding to the first state of the agent's trajectory
            frame: current frame
            maneuver: current maneuver in execution by the agent
            visible_region: region of the map which is visible to the ego vehicle
            search_nearby_lanes: whether to check nearby lanes if all goals have 0 probability
        """

        current_goals_probabilities = copy.deepcopy(goals_probabilities)

        if search_nearby_lanes is True:
            current_position = frame[agent_id].position
            nearby_lanes = self._scenario_map.lanes_within_angle(current_position, frame[agent_id].heading,
                                                                 np.pi / 4, drivable_only=True, max_distance=4)
            nearby_locations = [np.array(l.midline.interpolate(l.midline.project(Point(current_position))))
                                for l in nearby_lanes]

        norm_factor = 0.
        for goal_and_type, prob in current_goals_probabilities.goals_probabilities.items():
            try:
                goal = goal_and_type[0]

                if goal.reached(frame_ini[agent_id].position):
                    raise RuntimeError(f"Agent {agent_id} reached goal at start.")

                # 4. and 5. Generate optimum trajectory from initial point and smooth it
                if current_goals_probabilities.optimum_trajectory[goal_and_type] is None:
                    logger.debug("Generating optimum trajectory")
                    trajectories, plans = self.generate_trajectory(1, agent_id, frame_ini, goal,
                                                                    state_trajectory=None,
                                                                    visible_region=visible_region)
                    current_goals_probabilities.optimum_trajectory[goal_and_type] = trajectories[0]
                    current_goals_probabilities.optimum_plan[goal_and_type] = plans[0]

                opt_trajectory = current_goals_probabilities.optimum_trajectory[goal_and_type]

                # 7. and 8. Generate optimum trajectory from last observed point and smooth it
                all_trajectories, all_plans = self.generate_trajectory(
                    self._n_trajectories, agent_id, frame, goal, observed_trajectory,
                    visible_region=visible_region, debug=False)

                # 6. Calculate optimum reward
                current_goals_probabilities.optimum_reward[goal_and_type] = self._reward(opt_trajectory, goal)

                # For each generated possible trajectory to this goal
                for trajectory in all_trajectories:
                    # join the observed and generated trajectories
                    trajectory.insert(observed_trajectory)

                    # 9,10. calculate rewards, likelihood
                    reward = self._reward(trajectory, goal)
                    current_goals_probabilities.all_rewards[goal_and_type].append(reward)

                    reward_diff = self._reward_difference(opt_trajectory, trajectory, goal)
                    current_goals_probabilities.all_reward_differences[goal_and_type].append(reward_diff)

                # 11. Calculate likelihood
                likelihood = self._likelihood(opt_trajectory, all_trajectories[0], goal)

                # Calculate all trajectory probabilities
                current_goals_probabilities.trajectories_probabilities[goal_and_type] = \
                    self._trajectory_probabilities(current_goals_probabilities.all_rewards[goal_and_type])

                # Write additional goals probabilities fields
                current_goals_probabilities.all_trajectories[goal_and_type] = all_trajectories
                current_goals_probabilities.all_plans[goal_and_type] = all_plans
                current_goals_probabilities.current_trajectory[goal_and_type] = all_trajectories[0]
                current_goals_probabilities.reward_difference[goal_and_type] = \
                    current_goals_probabilities.all_reward_differences[goal_and_type][0]
                current_goals_probabilities.current_reward[goal_and_type] = \
                    current_goals_probabilities.all_rewards[goal_and_type][0]

            except RuntimeError as e:
                logger.debug(str(e))
                likelihood = 0.
                current_goals_probabilities.current_trajectory[goal_and_type] = None

            # update goal probabilities
            current_prob = current_goals_probabilities.goals_priors[goal_and_type] * likelihood
            # try to smooth the probability to avoid sharp changing
            if likelihood != 0:
                current_goals_probabilities.goals_probabilities[goal_and_type] = (current_goals_probabilities.goals_priors[
                                                                             goal_and_type] + current_prob) / 2
                norm_factor += (current_goals_probabilities.goals_priors[goal_and_type] + current_prob) / 2
            else:
                current_goals_probabilities.goals_probabilities[goal_and_type] = current_prob
                norm_factor += current_prob
            current_goals_probabilities.likelihood[goal_and_type] = likelihood


        # then divide prob by norm_factor to normalise
        for key, prob in current_goals_probabilities.goals_probabilities.items():
            try:
                current_goals_probabilities.goals_probabilities[key] = prob / norm_factor
            except ZeroDivisionError as e:
                logger.debug("All goals unreachable. Setting all probabilities to 0.")
                break

        # If no goal is reachable, try using a different nearby lane.
        all_goals_unreachable = np.sum(list(current_goals_probabilities.goals_probabilities.values())) == 0
        if all_goals_unreachable and search_nearby_lanes is True:

            for new_position in nearby_locations:
                # Re-run the function with the currect position set to a nearby location
                current_goals_probabilities = copy.deepcopy(goals_probabilities)
                new_frame = frame
                new_frame[agent_id].position = new_position
                current_goals_probabilities = self.update_goals_probabilities(current_goals_probabilities,
                                                                              observed_trajectory, agent_id, frame_ini,
                                                                              new_frame, maneuver,
                                                                              search_nearby_lanes = False)
                all_goals_unreachable = np.sum(list(current_goals_probabilities.goals_probabilities.values())) == 0
                if all_goals_unreachable == False:
                    break

        return current_goals_probabilities

    def generate_trajectory(self,
                            n_trajectories: int,
                            agent_id: int,
                            frame: Dict[int, ip.AgentState],
                            goal: ip.Goal,
                            state_trajectory: ip.Trajectory,
                            visible_region: ip.Circle = None,
                            n_resample=5,
                            debug=False) -> Tuple[List[ip.VelocityTrajectory], List[List[ip.MacroAction]]]:
        """Generates up to n possible trajectories from the current frame of an agent to the specified goal. """
        trajectories, plans = self._astar.search(agent_id, frame, goal,
                                                 self._scenario_map,
                                                 n_trajectories,
                                                 open_loop=True,
                                                 visible_region=visible_region,
                                                 debug=debug)
        if len(trajectories) == 0:
            raise RuntimeError(f"{goal} is unreachable")

        for trajectory in trajectories:
            if state_trajectory is None:
                trajectory.velocity[0] = frame[agent_id].speed  # Optimal case
            else:
                trajectory.velocity[0] = state_trajectory.velocity[-1]

            try:
                self._smoother.load_trajectory(trajectory)
                new_velocities = self._smoother.split_smooth()
            except RuntimeError as e:
                logger.debug(e)
                new_velocities = trajectory.velocity

            # Add linear sampling in the first n points and re-try smoothing if velocity smoothing failed
            initial_acc = np.abs(new_velocities[0] - new_velocities[1])
            if len(trajectory.velocity) > n_resample and initial_acc > frame[agent_id].metadata.max_acceleration:
                new_vels = ip.Maneuver.get_const_acceleration_vel(trajectory.velocity[0],
                                                                  trajectory.velocity[n_resample - 1],
                                                                  trajectory.path[:n_resample])
                trajectory.velocity[:n_resample] = new_vels

                self._smoother.load_trajectory(trajectory)
                new_velocities = self._smoother.split_smooth()
                trajectory.velocity = new_velocities

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

    def scenario_map(self):
        return self._scenario_map
