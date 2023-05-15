import copy
import pandas as pd
import os
import dill
import concurrent.futures
import argparse
import sys
import time

from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader
from igp2.core.goal import PointGoal
from igp2.recognition.goalrecognition import *
from igp2.recognition.astar import AStar
from igp2.core.cost import Cost
from igp2.core.results import *
from igp2.core.trajectory import StateTrajectory
from igp2.planlibrary.maneuver import Maneuver, SwitchLane
from igp2.planlibrary.macro_action import ChangeLane


def create_args():
    config_specification = argparse.ArgumentParser(description="""
This experiment will perform goal prediction for all agents and frames specified 
by a .csv file located in scripts/experiments/data/evaluation_set, following the
naming convention "scenario"_e"episode_no".csv, where episode_no is defined as the 
index of the recording ids defined in the json files. \n

A result binary will be generated and stored in scripts/experiments/data/results \n

Logs can be accessed in scripts/experiments/data/logs \n

Make sure to create these folders ahead of running the script.
     """, formatter_class=argparse.RawTextHelpFormatter)

    config_specification.add_argument('--num_workers', default="0",
                                      help="Number of parralel processes. Set 0 for auto", type=int)
    config_specification.add_argument('--output', default="experiment",
                                      help="Output .pkl filename", type=str)
    config_specification.add_argument('--tuning', default="0",
                                      help="0: runs default tuning parameters defined in the scenario JSON. \
                                      1: runs all cost parameters defined in the script.", type=int)
    config_specification.add_argument('--reward_scheme', default="1",
                                      help="0: runs the default reward scheme defined in the IGP2 paper. \
                                      1: runs alternate reward scheme defined in Cost.cost_difference_resampled().",
                                      type=int)
    config_specification.add_argument('--dataset', default="valid",
                                      help="valid: runs on the validation dataset, test: runs on the test dataset",
                                      type=str)

    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


def extract_goal_data(goals_data):
    """Creates a list of Goal objects from the .json scenario file data"""
    goals = []
    for goal_data in goals_data:
        point = Point(np.array(goal_data))
        goals.append(PointGoal(point, 2.))

    return goals


def read_and_process_data(scenario, episode_id):
    """Identifies which frames and agents to perform goal recognition for in episode,
    from a provided .csv file."""
    filename = str(scenario) + "_e" + str(episode_id) + ".csv"
    foldername = os.path.dirname(os.path.abspath(__file__)) + '/data/evaluation_set/'
    filename = foldername + filename
    data = pd.read_csv(filename)
    last_frame_id = None
    for index, row in data.iterrows():
        if last_frame_id is not None:
            if last_frame_id == row['frame_id'] and last_aid == row['agent_id']:
                data.drop(labels=index, axis=0, inplace=True)
        last_frame_id = row['frame_id']
        last_aid = row['agent_id']
    return data


def goal_recognition_agent(frames, recordingID, framerate, aid, data, goal_recognition: GoalRecognition,
                           goal_probabilities: GoalsProbabilities):
    """Computes the goal recognition results for specified agent at specified frames."""
    goal_probabilities_c = copy.deepcopy(goal_probabilities)
    result_agent = None
    for frame in frames:
        if aid in frame.dead_ids: frame.dead_ids.remove(aid)
    for _, row in data.iterrows():
        try:
            if result_agent == None: result_agent = AgentResult(row['true_goal'])
            frame_id = row['frame_id']
            frame_ini = row['initial_frame_id']
            agent_states = [frame.agents[aid] for frame in frames[0:frame_id - frame_ini + 1]]
            trajectory = StateTrajectory(framerate, frames[0].time, agent_states)
            t_start = time.perf_counter()
            goal_recognition.update_goals_probabilities(goal_probabilities_c, trajectory, aid,
                                                        frame_ini=frames[0].agents,
                                                        frame=frames[frame_id - frame_ini].agents, maneuver=None)
            t_end = time.perf_counter()
            result_agent.add_data((frame_id, copy.deepcopy(goal_probabilities_c), t_end - t_start, trajectory.path[-1]))
        except Exception as e:
            logger.error(f"Fatal in recording_id: {recordingID} for aid: {aid} at frame {frame_id}.")
            logger.error(f"Error message: {str(e)}")

    return aid, result_agent


def multi_proc_helper(arg_list):
    """Allows to pass multiple arguments as multiprocessing routine."""
    return goal_recognition_agent(arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4], arg_list[5],
                                  arg_list[6])


def run_experiment(cost_factors: Dict[str, float] = None, use_priors: bool = True, max_workers: int = None):
    """Run goal prediction in parralel for each agent, across all specified scenarios."""
    result_experiment = ExperimentResult()

    for SCENARIO in SCENARIOS:
        scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{SCENARIO}.xodr")
        data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", [DATASET])
        data_loader.load()

        # Scenario specific parameters
        SwitchLane.TARGET_SWITCH_LENGTH = data_loader.scenario.config.target_switch_length
        ChangeLane.CHECK_ONCOMING = data_loader.scenario.config.check_oncoming
        Trajectory.VELOCITY_STOP = 1.  # TODO make .json parameter
        Maneuver.NORM_WIDTH_ACCEPTABLE = 0.5
        Maneuver.LON_SWERVE_DISTANCE = 8

        if cost_factors is None:
            cost_factors = data_loader.scenario.config.cost_factors
        episode_ids = data_loader.scenario.config.dataset_split[DATASET]
        test_data = [read_and_process_data(SCENARIO, episode_id) for episode_id in episode_ids]
        goals_data = data_loader.scenario.config.goals
        if use_priors:
            goals_priors = data_loader.scenario.config.goals_priors
        else:
            goals_priors = None
        goals = extract_goal_data(goals_data)
        goal_probabilities = GoalsProbabilities(goals, priors=goals_priors)
        astar = AStar(next_lane_offset=0.25)
        cost = Cost(factors=cost_factors)
        ind_episode = 0
        for episode in data_loader:
            # episode specific parameters
            Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit
            cost.limits["velocity"] = episode.metadata.max_speed

            recordingID = episode.metadata.config['recordingId']
            framerate = episode.metadata.frame_rate
            logger.info(
                f"Starting experiment in scenario: {SCENARIO}, episode_id: {episode_ids[ind_episode]}, recording_id: {recordingID}")
            smoother = VelocitySmoother(vmin_m_s=Trajectory.VELOCITY_STOP, vmax_m_s=episode.metadata.max_speed, n=10,
                                        amax_m_s2=5, lambda_acc=10)
            goal_recognition = GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost,
                                               reward_as_difference=REWARD_AS_DIFFERENCE)
            result_episode = EpisodeResult(episode.metadata, episode_ids[ind_episode], cost_factors)

            # Prepare inputs for multiprocessing
            grouped_data = test_data[ind_episode].groupby('agent_id')
            args = []
            for aid, group in grouped_data:
                data = group.copy()
                frame_ini = data.initial_frame_id.values[0]
                frame_last = data.frame_id.values[-1]
                frames = episode.frames[frame_ini:frame_last + 1]
                arg = [frames, recordingID, framerate, aid, data, goal_recognition, goal_probabilities]
                args.append(copy.deepcopy(arg))

            # Perform multiprocessing
            results_agents = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            #with MockProcessPoolExecutor() as executor:
                results_agents = [executor.submit(multi_proc_helper, arg) for arg in args]
                for result_agent in concurrent.futures.as_completed(results_agents):
                    try:
                        result_episode.add_data(result_agent.result())
                    except Exception as e:
                        logger.error(f"Error during multiprocressing. Error message: {str(e)}")

            result_experiment.add_data((episode.metadata.config['recordingId'], copy.deepcopy(result_episode)))
            ind_episode += 1

    return result_experiment


def dump_results(objects, name: str):
    """Saves results binary"""
    filename = name + '.pkl'
    foldername = os.path.dirname(os.path.abspath(__file__)) + '/data/results/'
    filename = foldername + filename

    with open(filename, 'wb') as f:
        dill.dump(objects, f)


# Replace ProcessPoolExecutor with this for debugging without parallel execution
class MockProcessPoolExecutor():
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def submit(self, fn, *args, **kwargs):
        # execute functions in series without creating threads
        # for easier unit testing
        result = fn(*args, **kwargs)
        return result

    def shutdown(self, wait=True):
        pass


if __name__ == '__main__':

    global SCENARIOS
    global DATASET
    global TUNING
    global REWARD_AS_DIFFERENCE

    config = create_args()
    experiment_name = config['output']
    logger = setup_logging(level=logging.INFO, log_dir="scripts/experiments/data/logs", log_name=experiment_name)

    SCENARIOS = ["frankenberg", "bendplatz", "heckstrasse", "round"]

    DATASET = config["dataset"]
    if DATASET not in ('test', 'valid'):
        logger.error("Invalid dataset specified")
        sys.exit(1)

    TUNING = config["tuning"]
    if TUNING not in (0, 1):
        logger.error("Invalid tuning argument specified, use 0 or 1.")
        sys.exit(1)
    TUNING = bool(TUNING)

    REWARD_AS_DIFFERENCE = config["reward_scheme"]
    if REWARD_AS_DIFFERENCE not in (0, 1):
        logger.error("Invalid reward_scheme argument specified, use 0 or 1.")
        sys.exit(1)
    REWARD_AS_DIFFERENCE = bool(REWARD_AS_DIFFERENCE)

    max_workers = None if config['num_workers'] == 0 else config['num_workers']
    if max_workers is not None and max_workers <= 0:
        logger.error("Specify a valid number of workers or leave to default")
        sys.exit(1)

    results = []

    if TUNING:
        cost_factors_arr = []
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 1000, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 0.1, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 1, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 100, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        for idx, cost_factors in enumerate(cost_factors_arr):
            logger.info(f"Starting experiment {idx} with cost factors {cost_factors}.")
            t_start = time.perf_counter()
            result_experiment = run_experiment(cost_factors, use_priors=True, max_workers=max_workers)
            results.append(copy.deepcopy(result_experiment))
            t_end = time.perf_counter()
            logger.info(f"Experiment {idx} completed in {t_end - t_start} seconds.")
    else:
        logger.info(f"Starting experiment")
        t_start = time.perf_counter()
        result_experiment = run_experiment(cost_factors=None, use_priors=True, max_workers=max_workers)
        results.append(copy.deepcopy(result_experiment))
        t_end = time.perf_counter()
        logger.info(f"Experiment completed in {t_end - t_start} seconds.")

    dump_results(results, experiment_name)
    logger.info("All experiments complete.")
