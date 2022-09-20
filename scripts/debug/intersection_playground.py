from collections import defaultdict
import time
import igp2 as ip
import numpy as np
import random
import logging
import carla
import matplotlib.pyplot as plt
from scripts.experiments.scenarios.util import parse_args, generate_random_frame

ip.setup_logging()
logger = logging.getLogger(__name__)
args = parse_args()

# Set run parameters here
seed = args.seed
max_speed = args.max_speed
ego_id = 0
n_simulations = args.n_sim
fps = args.fps  # Simulator frequency
T = args.period  # MCTS update period

random.seed(seed)
np.random.seed(seed)
np.seterr(divide="ignore")
ip.Maneuver.MAX_SPEED = max_speed

scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/intersection.xodr")
veh1_spawn_box = ip.Box(np.array([-160.0, -102.5]), 10, 3.5, 0.0)
veh2_spawn_box = ip.Box(np.array([80, -97.5]), 10, 3.5, 0.0)
veh3_spawn_box = ip.Box(np.array([60, -204]), 3.5, 10, 0.25*np.pi)
frame = generate_random_frame(0, scenario_map,
                              [(veh1_spawn_box, (0.0, 0.0)),
                               (veh2_spawn_box, (0.0, 0.0)),
                               (veh3_spawn_box, (0.0, 0.0))])
goals = {
    0: ip.BoxGoal(ip.Box(np.array([-140.0, 0.0]), 5, 7, 0.25*np.pi)),
    1: ip.BoxGoal(ip.Box(np.array([30, -181.0]), 5, 7, 0.25*np.pi)),
    2: ip.BoxGoal(ip.Box(np.array([-140.0, 0.0]), 5, 7, 0.25*np.pi))
}
# ip.plot_map(scenario_map)
# plt.plot(*list(zip(*veh1_spawn_box.boundary)))
# plt.plot(*list(zip(*veh2_spawn_box.boundary)))
# plt.plot(*list(zip(*veh3_spawn_box.boundary)))
# for aid, state in frame.items():
#     plt.plot(*state.position, marker="x")
#     plt.text(*state.position, aid)
# for goal in goals.values():
#     plt.plot(*list(zip(*goal.box.boundary)), c="g")
# plt.show()

carla_sim = ip.carla.CarlaSim(xodr="scenarios/maps/intersection.xodr")
for actor in carla_sim.world.get_actors().filter("*vehicle*"):
    actor.destroy()
carla_sim.spectator.set_location(carla.Location(-50, 100, 25.0))
carla_sim.world.tick()

blueprint_library = carla_sim.world.get_blueprint_library()
blueprints = {
    0: blueprint_library.find('vehicle.audi.a2'),
    1: blueprint_library.find('vehicle.bmw.grandtourer'),
    2: blueprint_library.find('vehicle.ford.mustang')
}

agents = {}
agents_meta = ip.AgentMetadata.default_meta_frame(frame)
for aid in frame.keys():
    goal = goals[aid]
    agents[aid] = ip.TrafficAgent(aid, frame[aid], goal, fps)
    carla_sim.add_agent(agents[aid], None, blueprint=blueprints[aid])

vels = defaultdict(list)
target_speeds = {}
for t in range(40 * 20):
    obs, _ = carla_sim.step()
    frame = obs.frame
    for aid, state in frame.items():
        vels[aid].append(state.speed)
        if carla_sim.agents[aid] is not None:
            target_speeds[aid] = carla_sim.agents[aid].target_speeds
    # time.sleep(1 / carla_sim.fps)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
for i, (aid, agent_wrapper) in enumerate(carla_sim.agents.items()):
    avels = vels[aid]
    color = colors[i % len(colors)]
    plt.plot(range(len(avels)), avels, label=f"{aid} State Speed", c=color)
    if len(avels) >= len(target_speeds[aid]):
        plt.plot(range(len(target_speeds[aid])), target_speeds[aid], "--", c=color, label=f"{aid} Target Speed")
    else:
        plt.plot(range(len(avels)), target_speeds[aid][:len(avels)], "--", c=color, label=f"{aid} Target Speed")
    plt.legend()
    plt.show()