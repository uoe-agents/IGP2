import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas

from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver
from igp2 import setup_logging
from igp2.data.data_loaders import InDDataLoader
from igp2.trajectory import Trajectory, StateTrajectory

# This script computes the maximum values encountered in the ind dataset for each 
# quantities used to compute rewards associated with trajectories. It will then 
# generate a file containing the results in scripts/experiments/data/limit_values.csv

def remove_outliers(array, max_deviations: int = 3):
    mean = np.mean(array)
    std = np.std(array)
    distance_mean = abs(array - mean)
    not_outlier_i = distance_mean < max_deviations * std
    return array[not_outlier_i]

def save_file(limit_values: dict()):
    df = pandas.DataFrame(limit_values, index=[0])
    filename = os.path.dirname(os.path.abspath(__file__))  + '/data/limit_values.csv'
    df.to_csv(filename, index=False)

def plot_data(traj_properties: dict(), plot_col: int = 2):
    plot_row = int(math.ceil(len(traj_properties.keys())/2))
    fig, axs = plt.subplots(plot_row, plot_col)
    axs = axs.ravel()

    i = 0
    for attribute, inner_dict in traj_properties.items():
        colors = list("rgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmyk")
        for key, array in inner_dict.items():
            minv = array[:,0]
            maxv = array[:,1]
            color = colors.pop()
            x = [j for j in range(0, len(minv))]
            axs[i].scatter(x, minv,color=color, label=str(key), marker='.')
            axs[i].scatter(x, maxv,color=color, marker='.')
        axs[i].legend()
        axs[i].set(xlabel="AgentId", ylabel=attribute)
        i += 1

    plt.show()


SCENARIOS = ["heckstrasse", "frankenberg", "bendplatz"]
GENERATE_PLOT = True
SAVE_FILE = True

if __name__ == '__main__':
    setup_logging()

    traj_properties = {"velocity": dict(), "acceleration": dict(), "jerk": dict(), "angular_velocity": dict(),
                         "angular_acceleration": dict(), "curvature": dict()}

    for SCENARIO in SCENARIOS:
        print("Loading Scenario: ", SCENARIO)
        data_loader = InDDataLoader(f"scenarios/configs/{SCENARIO}.json", ["train", "valid"])
        data_loader.load()
        for episode in data_loader:

            recordingId = episode.metadata.config['recordingId']
            for attribute, inner_dict in traj_properties.items():
                inner_dict[recordingId] = np.empty((0,2), float)

            # Iterate over all agents and their full trajectories
            for agent_id, agent in episode.agents.items():
                agent_trajectory = agent.trajectory

                for attribute, inner_dict in traj_properties.items():
                    class_property = getattr(Trajectory, attribute)
                    minmax = np.array([[class_property.fget(agent.trajectory).min(), class_property.fget(agent.trajectory).max()]])
                    inner_dict[recordingId] = np.append(inner_dict[recordingId], minmax, axis = 0)

    limit_values = dict()
    for attribute, inner_dict in traj_properties.items():
        joined_values = np.empty((0,2), float)
        for key, array in inner_dict.items():
            joined_values = np.concatenate((joined_values, array))
        minvs_no_outliers = remove_outliers(joined_values[:,0])
        maxvs_no_outliers = remove_outliers(joined_values[:,1])
        minkey = "min_" + attribute
        maxkey = "max_" + attribute
        limit_values[minkey] = minvs_no_outliers.min()
        limit_values[maxkey] = maxvs_no_outliers.max()

    if SAVE_FILE: save_file(limit_values)
    if GENERATE_PLOT : plot_data(traj_properties)
