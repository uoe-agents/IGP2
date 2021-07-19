import dill
import numpy as np
import os
import matplotlib.pyplot as plt
import json

def load_results(filenames):

    foldername = os.path.dirname(os.path.abspath(__file__))  + '/data/cost_tuning/'
    results = []
    for filename in filenames:
        filename = foldername + filename
        with open(filename, 'rb') as f:
            results += dill.load(f)

    return results

def dump_results(objects, name : str):
    filename = name + '.pkl'
    foldername = os.path.dirname(os.path.abspath(__file__))  + '/data/cost_tuning/'
    filename = foldername + filename

    with open(filename, 'wb') as f:
        dill.dump(objects, f)

filenames = ['validset_allscenarios_swerve_distance_01.pkl']

results = load_results(filenames)

#to_save = [results[3]]
#dump_results(to_save, 'validset_all_scenarios')

#indices = [1,2,3,4]
indices = []
for i in sorted(indices, reverse=True):
    del results[i]

REMOVE_UNCOMPLETED_PATHS = False
REMOVE_UNFEASIBLE_PATHS = False
EXPERIMENT = 'valid'

#preprocessing: remove agents with less than 11 points and/or agents with unfeasible true goals
if REMOVE_UNCOMPLETED_PATHS or REMOVE_UNFEASIBLE_PATHS:
    print("Preprocessing...")
    for ind, result in enumerate(results):
        print(f"Experiment: {ind}")
        for episode_result in result.data:
            print(f"Recording ID: {episode_result[0]}")
            data_dict = dict(episode_result[1].data)
            for agent_result in episode_result[1].data:
                if REMOVE_UNCOMPLETED_PATHS and len(agent_result[1].true_goal_probability) != 11:
                    print(f"Agent {agent_result[0]} has only {len(agent_result[1].true_goal_probability)} frames.")
                    print("Removing agent.")
                    data_dict.pop(agent_result[0])
                remove = False
                for prob in agent_result[1].zero_probability:
                    if prob : remove = True
                if REMOVE_UNFEASIBLE_PATHS and remove :
                    print(f"Agent: {agent_result[0]} has only unfeasible path to goals")
                    print(f"Removing agent.") 
                    try:
                        data_dict.pop(agent_result[0])
                    except KeyError:
                        print("Key Error, agent already removed")
            episode_result[1].data = list(tuple(data_dict.items()))

print("---------------------------------")

#experiment5: return agents that have very large trajectory duration:
for episode_result in results[0].data:
    if EXPERIMENT == 'valid':
        if episode_result[0] == 26: scenario ="Frankenberg"
        elif episode_result[0] == 7: scenario= "Bendplatz"
        elif episode_result[0] == 31: scenario = "Heckstrasse"
        elif episode_result[0] == 5 or episode_result[0] == 15: scenario = "Round"
        else: scenario = "Unknown"
    if EXPERIMENT == 'test':
        if episode_result[0] == 23 and episode_result[1].id == 5: scenario ="Frankenberg"
        elif episode_result[0] == 15 and episode_result[1].id == 8: scenario= "Bendplatz"
        elif episode_result[0] == 30 and episode_result[1].id == 0: scenario = "Heckstrasse"
        elif episode_result[0] == 6 and episode_result[1].id == 4 or episode_result[0] == 23 and episode_result[1].id == 21: scenario = "Round"
        else: scenario = "Unknown"

    print(f"SCENARIO : {scenario}, Recording ID: {episode_result[0]}")

    for agent_result in episode_result[1].data:
        for ind, goal_prob_o in enumerate(agent_result[1].data):
            curr_traj = list(goal_prob_o[1].current_trajectory.values())[agent_result[1].true_goal]
            opt_traj = list(goal_prob_o[1].optimum_trajectory.values())[agent_result[1].true_goal]
            if opt_traj is not None and curr_traj is not None:
                if curr_traj.duration > 30:
                    print(f"Agent {agent_result[0]}, frame {goal_prob_o[0]} has curr / opt duration of {curr_traj.duration} / {opt_traj.duration} s. \
                    Goal accurate: {agent_result[1].goal_accuracy[ind]}")

print("--------------------------------------")

#experiment0: print agents and frames causing spikes
all_spikes = {"Frankenberg" : [7], "Bendplatz" : [3,4,5,6,7], "Heckstrasse" : [4,7], "Round" : [2]}

for episode_result in results[0].data:
    if EXPERIMENT == 'valid':
        if episode_result[0] == 26: scenario ="Frankenberg"
        elif episode_result[0] == 7: scenario= "Bendplatz"
        elif episode_result[0] == 31: scenario = "Heckstrasse"
        elif episode_result[0] == 5 or episode_result[0] == 15: scenario = "Round"
        else: scenario = "Unknown"
    if EXPERIMENT == 'test':
        if episode_result[0] == 23 and episode_result[1].id == 5: scenario ="Frankenberg"
        elif episode_result[0] == 15 and episode_result[1].id == 8: scenario= "Bendplatz"
        elif episode_result[0] == 30 and episode_result[1].id == 0: scenario = "Heckstrasse"
        elif episode_result[0] == 6 and episode_result[1].id == 4 or episode_result[0] == 23 and episode_result[1].id == 21: scenario = "Round"
        else: scenario = "Unknown"

    print(f"SCENARIO : {scenario}, Recording ID: {episode_result[0]}")

    spikes = all_spikes[scenario]
    culprits = []
    for spike in spikes:
        for agent_result in episode_result[1].data:
            if agent_result[1].zero_probability[spike] :#and (not agent_result[1].zero_probability[spike - 1] and not agent_result[1].zero_probability[spike + 1]):
                culprits.append([agent_result[1].data[spike][0],agent_result[0], agent_result[1].true_goal, agent_result[1].data[spike][3]])
    
    for culprit in sorted(culprits):
        print(f"Spike at frame {culprit[0]} caused by agent {culprit[1]} with true goal {culprit[2]} at position {culprit[3]}")

print("--------------------------------------")

#experiment1: count the number of times prob is 0 in raw data
for episode_result in results[0].data:
    print(f"Recording ID: {episode_result[0]}")
    print(f"Zero true goal fraction: {episode_result[1].zero_probability.mean()}")
    n_goals = len(episode_result[1].data[0][1].data[0][1].goals_probabilities)
    count_0_prob = np.zeros(n_goals)
    count_total = 0
    for agent_result in episode_result[1].data:
        for zero_prob in agent_result[1].zero_probability:
            if zero_prob == 1 : 
                count_0_prob[agent_result[1].true_goal] += 1
                count_total += 1
    prob_0 = count_0_prob / count_total
    for i in range(0, len(prob_0)):
        print(f"Goal {i} is responsible for {prob_0[i] * 100} % of zero true goal fraction")

print("---------------------------------")

#experiment2: calculate average true goal probability for each episode
fig, axs = plt.subplots(5)
axs = axs.ravel()

k = 0
for ind , result in enumerate(results):
    print(f"Experiment {ind}, cost factors: {json.dumps(result.data[0][1].cost_factors)}")
    print(f"Mean true goal probability: {result.true_goal_probability.mean()}")
    i = 0
    for episode_result in result.data:
        label = json.dumps(episode_result[1].cost_factors)
        #label = "testset_data"
        if EXPERIMENT == 'valid':
            if episode_result[0] == 26: scenario ="Frankenberg"
            elif episode_result[0] == 7: scenario= "Bendplatz"
            elif episode_result[0] == 31: scenario = "Heckstrasse"
            elif episode_result[0] == 5 or episode_result[0] == 15: scenario = "Round"
            else: scenario = "Unknown"
        if EXPERIMENT == 'test':
            if episode_result[0] == 23 and episode_result[1].id == 5: scenario ="Frankenberg"
            elif episode_result[0] == 15 and episode_result[1].id == 8: scenario= "Bendplatz"
            elif episode_result[0] == 30 and episode_result[1].id == 0: scenario = "Heckstrasse"
            elif episode_result[0] == 6 and episode_result[1].id == 4 or episode_result[0] == 23 and episode_result[1].id == 21: scenario = "Round"
            else: scenario = "Unknown"
        title = "Scenario " + str(scenario) + " Recording ID " + str(episode_result[0])
        axs[i].set_title(title)
        if k == 0 : axs[i].plot(episode_result[1].zero_probability, 'k', label = 'Unfeasible path fractions')
        axs[i].plot(episode_result[1].true_goal_probability, label=label)
        low_std = np.clip(episode_result[1].true_goal_probability-episode_result[1].true_goal_ste, 0, 1)
        high_std = np.clip(episode_result[1].true_goal_probability+episode_result[1].true_goal_ste, 0, 1)
        axs[i].fill_between(range(11), low_std, high_std, alpha = 0.2)
        axs[i].set(ylabel="True Goal Probability")
        axs[i].set_ylim(0, 1)
        i+=1
    k += 1
axs[-1].set(xlabel="Trajectory fraction")
axs[-1].legend(loc="lower center", bbox_to_anchor=(0.5, -1.6))
fig.subplots_adjust(bottom=0.25)
plt.show()

print("---------------------------------")

#experiment3: calculate goal_accuracy for each episode
fig, axs = plt.subplots(5)
axs = axs.ravel()
k = 0
for result in results:
    print(f"Experiment {ind}, cost factors: {json.dumps(result.data[0][1].cost_factors)}")
    print(f"Mean goal accuracy: {result.goal_accuracy.mean()}")
    i = 0
    for episode_result in result.data:
        label = json.dumps(episode_result[1].cost_factors)
        #label = "testset_data"
        if EXPERIMENT == 'valid':
            if episode_result[0] == 26: scenario ="Frankenberg"
            elif episode_result[0] == 7: scenario= "Bendplatz"
            elif episode_result[0] == 31: scenario = "Heckstrasse"
            elif episode_result[0] == 5 or episode_result[0] == 15: scenario = "Round"
            else: scenario = "Unknown"
        if EXPERIMENT == 'test':
            if episode_result[0] == 23 and episode_result[1].id == 5: scenario ="Frankenberg"
            elif episode_result[0] == 15 and episode_result[1].id == 8: scenario= "Bendplatz"
            elif episode_result[0] == 30 and episode_result[1].id == 0: scenario = "Heckstrasse"
            elif episode_result[0] == 6 and episode_result[1].id == 4 or episode_result[0] == 23 and episode_result[1].id == 21: scenario = "Round"
            else: scenario = "Unknown"
        title = "Scenario " + str(scenario) + " Recording ID " + str(episode_result[0])
        axs[i].set_title(title)
        if k == 0 : axs[i].plot(episode_result[1].zero_probability, 'k', label = 'Unfeasible path fractions')
        axs[i].plot(episode_result[1].goal_accuracy, label=label)
        low_std = np.clip(episode_result[1].goal_accuracy-episode_result[1].goal_accuracy_ste, 0, 1)
        high_std = np.clip(episode_result[1].goal_accuracy+episode_result[1].goal_accuracy_ste, 0, 1)
        axs[i].fill_between(range(11), low_std, high_std, alpha = 0.2)
        axs[i].set(ylabel="Goal Accuracy")
        axs[i].set_ylim(0, 1)
        i+=1
    k += 1
axs[-1].set(xlabel="Trajectory fraction")
axs[-1].legend(loc="lower center", bbox_to_anchor=(0.5, -1.6))
fig.subplots_adjust(bottom=0.25)
plt.show()


print("Done")