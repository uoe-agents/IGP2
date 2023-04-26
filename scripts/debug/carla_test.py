from collections import defaultdict

import carla
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
from igp2.carlasim import CarlaSim
from igp2.carlasim.util import get_speed
from igp2 import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

seed = 0
random.seed(seed)
np.random.seed(seed)

client = CarlaSim(map_name="Town01")
for actor in client.world.get_actors().filter("*vehicle*"):
    actor.destroy()
client.world.tick()

tm = client.get_traffic_manager()
tm.set_agents_count(10)
tm.set_spawn_filter("vehicle.audi.a2")
tm.update(client)

# agent_wrapper = list(client.agents.values())[0]
# client.spectator.set_location(carla.Location(agent_wrapper.state.position[0], -agent_wrapper.state.position[1], 5.0))

vels = defaultdict(list)
for t in range(60 * 20):
    obs, _ = client.step()
    frame = obs.frame
    for aid, state in frame.items():
        vels[aid].append(state.speed)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
for i, (aid, agent_wrapper) in enumerate(client.agents.items()):
    avels = vels[aid]
    color = colors[i % len(colors)]
    plt.plot(range(len(avels)), avels, label=f"{aid} State Speed", c=color)
    plt.plot(range(len(avels)), agent_wrapper.target_speeds[:len(avels)], "--", c=color, label=f"{aid} Target Speed")
    plt.legend()
    plt.show()


# client.load_world('Town01')
# settings = world.get_settings()
# settings.fixed_delta_seconds = 1 / 20
# settings.synchronous_mode = True
# world.apply_settings(settings)

# map = world.get_map()
# blueprint_library = world.get_blueprint_library()
# spawns = map.get_spawn_points()
# blueprint = blueprint_library.find('vehicle.audi.a2')
#
# actor = world.spawn_actor(blueprint, spawns[0])
# # actor.set_target_velocity(carla.Vector3D(0, 5, 0))
# world.tick()

# while True:
#     control = carla.VehicleControl(throttle=0.5)
#     commands = []
#     vel = actor.get_velocity()
#     speed = np.sqrt(vel.x ** 2 + vel.y ** 2)
#     actor.set_transform(spawns[0])
#     if speed >= 5:
#         break
#     command = carla.command.ApplyVehicleControl(actor, control)
#     commands.append(command)
#     client.apply_batch_sync(commands)
#     world.tick()
#
# while True:
#     world.tick()
#     actor_list = world.get_actors()
#     vehicle_list = actor_list.filter("*vehicle*")
#     for v in vehicle_list:
#         print(v.get_velocity())
#     # actor.set_target_velocity(carla.Vector3D(0, 5, 0))
#     client.apply_batch_sync([carla.command.ApplyVehicleControl(actor, carla.VehicleControl(throttle=0.3))])
