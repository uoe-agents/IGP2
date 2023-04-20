import carla

import igp2 as ip
import random
import numpy as np
import matplotlib.pyplot as plt
import logging

from scripts.experiments.scenarios.util import parse_args, generate_random_frame

if __name__ == '__main__':
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

    # Set randomised spawn parameters here
    ego_spawn_box = ip.Box(np.array([-80.0, -1.8]), 10, 3.5, 0.0)
    ego_vel_range = (5.0, max_speed)
    veh1_spawn_box = ip.Box(np.array([-70.0, 1.7]), 10, 3.5, 0.0)
    veh1_vel_range = (5.0, max_speed)
    veh2_spawn_box = ip.Box(np.array([-18.34, -25.5]), 3.5, 10, 0.0)
    veh2_vel_range = (5.0, max_speed)

    # Vehicle goals
    goals = {
        ego_id: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0)),
        ego_id + 1: ip.BoxGoal(ip.Box(np.array([-22, -25.5]), 3.5, 5, 0.0)),
        ego_id + 2: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0))
    }

    scenario_path = "scenarios/maps/scenario1.xodr"
    scenario_map = ip.Map.parse_from_opendrive(scenario_path)

    frame = generate_random_frame(ego_id,
                                  scenario_map,
                                  [(ego_spawn_box, ego_vel_range),
                                   (veh1_spawn_box, veh1_vel_range),
                                   (veh2_spawn_box, veh2_vel_range)])

    ip.plot_map(scenario_map, markings=True, midline=True)
    plt.plot(*list(zip(*ego_spawn_box.boundary)))
    plt.plot(*list(zip(*veh1_spawn_box.boundary)))
    plt.plot(*list(zip(*veh2_spawn_box.boundary)))
    for aid, state in frame.items():
        plt.plot(*state.position, marker="x")
        plt.text(*state.position, aid)
    for goal in goals.values():
        plt.plot(*list(zip(*goal.box.boundary)), c="g")
    plt.gca().add_patch(plt.Circle(frame[0].position, 100, color='b', fill=False))
    plt.show()

    cost_factors = {"time": 0.1, "velocity": 0.0, "acceleration": 0.1, "jerk": 0., "heading": 0.0,
                    "angular_velocity": 0.1, "angular_acceleration": 0.1, "curvature": 0.0, "safety": 0.}
    reward_factors = {"time": 1.0, "jerk": -0.1, "angular_acceleration": -0.2, "curvature": -0.1}
    carla_sim = ip.carla.CarlaSim(xodr=scenario_path, carla_path=args.carla_path)

    agents = {}
    agents_meta = ip.AgentMetadata.default_meta_frame(frame)
    for aid in frame.keys():
        goal = goals[aid]

        if aid == ego_id:
            agents[aid] = ip.MCTSAgent(agent_id=aid,
                                       initial_state=frame[aid],
                                       t_update=T,
                                       scenario_map=scenario_map,
                                       goal=goal,
                                       cost_factors=cost_factors,
                                       reward_factors=reward_factors,
                                       fps=fps,
                                       n_simulations=n_simulations,
                                       view_radius=100,
                                       store_results="all")
            carla_sim.add_agent(agents[aid], "ego")
            carla_sim.spectator.set_location(
                carla.Location(frame[aid].position[0], -frame[aid].position[1], 5.0))
        else:
            agents[aid] = ip.TrafficAgent(aid, frame[aid], goal, fps)
            carla_sim.add_agent(agents[aid], None)

    observations = []
    actions = []
    colors = ['r', 'g', 'b', 'y', 'k']
    for t in range(500):
        obs, acts = carla_sim.step()
        observations.append(obs)
        actions.append(acts)

        logger.info(f'Step {t}')
        logger.info('Vehicle actions were:')
        for aid, act in acts.items():
            logger.info(f'Throttle: {act.throttle}; Steering: {act.steer}')

        xs = np.arange(t + 1)
        fix, axes = plt.subplots(1, 3)

        logger.info(f'Vehicle status:')
        for aid, agent in obs.frame.items():
            c = colors[aid % len(colors)]
            logger.info(f'Agent {aid}: v={agent.speed} @ ({agent.position[0]:.2f}, {agent.position[1]:.2f}); theta={agent.heading}')

            # Plot observed velocity
            ax = axes[0]
            vels = np.array([ob.frame[aid].speed for ob in observations])
            ax.plot(xs, vels, c=c)
            ax.set_title('Velocity')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Velocity (m/s)')

            # Plot throttle
            ax = axes[1]
            throttles = np.array([act[aid].throttle for act in actions])
            ax.plot(xs, throttles, c=c)
            ax.set_title('Throttle')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Throttle')

        plt.show()
