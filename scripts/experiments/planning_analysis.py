import os

import dill
from matplotlib import pyplot as plt

from igp2.results import PlanningResult


def load_results(filenames):

    foldername = os.path.dirname(os.path.abspath(__file__))  + '/data/planning_results/'
    for filename in filenames:
        filename = foldername + filename
        with open(filename, 'rb') as f:
            results = dill.load(f)

    return results

filenames = ['test_result.pkl']

results = load_results(filenames)

scenario_map = results.scenario_map

MCTS_result = results.results[0]
MCTS_tree = MCTS_result.tree.tree
key = ("Root",)
run_result = MCTS_tree[key].run_result

plt.show()

for t in range(0, len(run_result.agents[0].trajectory_cl.states)):
    run_result.plot(t, scenario_map)
    plt.show()

Print("Done")