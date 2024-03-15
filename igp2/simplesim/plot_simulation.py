from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from igp2.simplesim.simulation import Simulation
from igp2.agents.mcts_agent import MCTSAgent
from igp2.agents.macro_agent import MacroAgent
from igp2.agents.trajectory_agent import TrajectoryAgent
from igp2.agents.agent import Agent
from igp2.core.vehicle import Action
from igp2.opendrive.plot_map import plot_map

# -----------Simulation plotting functions---------------------


def plot_simulation(simulation: Simulation, axes: plt.Axes = None, debug: bool = False, map_plotter = None) \
        -> (plt.Figure, plt.Axes):
    """ Plot the current agents and the road layout for visualisation purposes.

    Args:
        simulation: The simulation to plot.
        axes: Axis to draw on
        debug: If True then plot diagnostic information.
        map_plotter: Function overriden default method to map road layout
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig = plt.gcf()
    fig.suptitle(f"T={simulation.t}")

    color_map_ego = plt.cm.get_cmap('Reds')
    color_map_non_ego = plt.cm.get_cmap('Blues')
    color_ego = 'r'
    color_non_ego = 'b'
    color_bar_non_ego = None

    ax = axes[0]
    if map_plotter is not None:
        map_plotter(simulation.scenario_map, markings=True, hide_road_bounds_in_junction=True, ax=ax)
    else:
        plot_map(simulation.scenario_map, markings=True, hide_road_bounds_in_junction=True, ax=ax)
    for agent_id, agent in simulation.agents.items():
        if agent is None or not agent.alive:
            continue

        if isinstance(agent, MCTSAgent):
            color = color_ego
            color_map = color_map_ego
        else:
            color = color_non_ego
            color_map = color_map_non_ego

        path = None
        velocity = None
        if isinstance(agent, MacroAgent) and agent.current_macro is not None:
            path = agent.current_macro.current_maneuver.trajectory.path
            velocity = agent.current_macro.current_maneuver.trajectory.velocity
        elif isinstance(agent, TrajectoryAgent) and agent.trajectory is not None:
            path = agent.trajectory.path
            velocity = agent.trajectory.velocity

        agent_plot = None
        if path is not None and velocity is not None:
            agent_plot = ax.scatter(path[:, 0], path[:, 1], c=velocity, cmap=color_map, vmin=-4, vmax=20, s=8)

        vehicle = agent.vehicle
        pol = plt.Polygon(vehicle.boundary, color=color)
        ax.add_patch(pol)
        ax.text(*agent.state.position, agent_id)
        if isinstance(agent, MCTSAgent) and agent_plot is not None:
            plt.colorbar(agent_plot, ax=ax)
            plt.text(0, 0.1, 'Current Velocity: ' + str(agent.state.speed), horizontalalignment='left',
                     verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0, 0.05, 'Current Macro Action: ' + agent.current_macro.__repr__(), horizontalalignment='left',
                     verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0, 0, 'Current Maneuver: ' + agent.current_macro.current_maneuver.__repr__(),
                     horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

            # Plot goals
            for gid, goal in enumerate(agent.possible_goals):
                loc = goal.center
                ax.plot(*loc, "ro")
                ax.plot(*loc, "kx")
                ax.text(*loc, gid)

            # Plot goal probabilities
            plot_predictions(agent, simulation.agents, axes[1], debug)
        elif isinstance(agent, TrajectoryAgent) and color_bar_non_ego is None:
            color_bar_non_ego = plt.colorbar(agent_plot, location="left")
        plt.text(*agent.state.position, agent_id)

    if debug:
        plot_diagnostics(simulation.agents, simulation.actions)

    return fig, axes


def plot_diagnostics(agents: Dict[int, Agent], actions: Dict[int, List[Action]]) -> (plt.Figure, plt.Axes):
    # attributes = ["velocity", "heading", "angular_velocity"]
    attributes = ["velocity", "acceleration", "jerk"]
    n_agents = len(agents)
    n_attributes = len(attributes)
    subplot_w = 5

    # Plot observations
    fig, axes = plt.subplots(n_agents, n_attributes,
                             figsize=(n_attributes * subplot_w, n_agents * subplot_w))
    if n_agents < 2:
        axes = axes[None, :]
    for i, (aid, agent) in enumerate(agents.items()):
        if agent is None:
            continue
        agent.trajectory_cl.calculate_path_and_velocity()
        ts = agent.trajectory_cl.times
        for j, attribute in enumerate(attributes):
            ax = axes[i, j]
            ys = getattr(agent.trajectory_cl, attribute)
            ys = np.round(ys, 4)
            ax.plot(ts, ys, label="Observed")
            ax.scatter(ts, ys, s=5)

            # Plot target velocities
            if attribute == "velocity":
                ys = [action.target_speed for action in actions[aid]]
                ys = [ys[0]] + ys
                ax.plot(ts, ys, c="red", label="Target")
                ax.scatter(ts, ys, s=5, c="red")
            axes[0, j].set_title(attribute)
            plot_maneuvers(agent, ax)
        axes[i, 0].set_ylabel(f"Agent {aid}")
        axes[i, 0].legend()
    fig.tight_layout()
    return fig, axes


def plot_maneuvers(agent: Agent, ax: plt.Axes) -> plt.Axes:
    man_list = np.array([state.maneuver for state in agent.trajectory_cl.states])
    man_list[0] = man_list[1]
    ts = agent.trajectory_cl.times
    colors = ["red", "blue", "green"]
    t_start = 0
    i = 0
    t_max = len(man_list)
    for t_end, (a, b) in enumerate(zip(man_list[:-1], man_list[1:]), 1):
        if a != b:
            ax.axvspan(ts[t_start], ts[t_end], facecolor=colors[i % len(colors)], alpha=0.2)
            ax.annotate(a, xy=((t_start + 0.5 * (t_end - t_start)) / t_max, 0.0), rotation=-45,
                        xycoords='axes fraction', fontsize=10, xytext=(-20, 5), textcoords='offset points')
            t_start = t_end
            i += 1
    if ts[t_start] != ts[-1]:
        ax.axvspan(ts[t_start], ts[-1], facecolor=colors[i % len(colors)], alpha=0.2)
        ax.annotate(a, xy=((t_start + 0.5 * (t_max - t_start)) / t_max, 0.0), rotation=-45,
                    xycoords='axes fraction', fontsize=10, xytext=(-30, 5), textcoords='offset points')
    return ax


def plot_predictions(ego_agent: MCTSAgent,
                     agents: Dict[int, Agent],
                     ax: plt.Axes,
                     debug: bool = False) -> plt.Axes:
    x, y = 0., 1.
    dx, dy = 0.5, 0.05
    ax.text(x, y, "Goal Prediction Probabilities", fontsize="large")
    y -= 2 * dy
    for i, (aid, goals_probs) in enumerate(ego_agent.goal_probabilities.items()):
        if i > 0 and i % 2 == 0:
            x += dx
            y = 0.9
        ax.text(x, y, f"Agent {aid}:", fontsize="medium")
        y -= dy
        for gid, (goal, gp) in enumerate(goals_probs.goals_probabilities.items()):
            if np.isclose(gp, 0.0):
                continue
            ax.text(x, y,
                    rf"   $P(g^{aid}_{gid}|s^{aid}_{{1:{ego_agent.trajectory_cl.states[-1].time}}})={gp:.3f}$:")
            y -= dy
            for tid, tp in enumerate(goals_probs.trajectories_probabilities[goal]):
                ax.text(x, y, rf"       $P(\hat{{s}}^{{{aid}, {tid}}}_{{1:n}}|g^{aid}_{gid})={tp:.3f}$")
                y -= dy
        y -= dy
    ax.axis("off")

    # Plot prediction trajectories
    if debug:
        attribute = "velocity"
        n_agents = max(2, len(agents) - 1)  # To make sure indexing works later on, at least 2 agents
        n_goals = len(ego_agent.possible_goals)
        subplot_w = 5

        fig, axes = plt.subplots(n_agents, n_goals,
                                 figsize=(n_goals * subplot_w, n_agents * subplot_w,))
        i = 0
        for aid, agent in agents.items():
            if agent.agent_id == ego_agent.agent_id:
                continue
            axes[i, 0].set_ylabel(f"Agent {aid}")
            probs = ego_agent.goal_probabilities[aid]
            for gid, goal in enumerate(probs.goals_probabilities):
                axes[0, gid].set_title(f"{goal[0]}", fontsize=10)
                ax = axes[i, gid]
                opt_trajectory = probs.optimum_trajectory[goal]
                if probs.all_trajectories[goal]:
                    trajectory = probs.all_trajectories[goal][0]
                    ax.plot(opt_trajectory.times, getattr(opt_trajectory, attribute), "r", label="Optimal")
                    ax.plot(trajectory.times, getattr(trajectory, attribute), "b", label="Observed")
            axes[i, 0].legend()
            i += 1
        fig.suptitle(attribute)
        fig.tight_layout()
    return ax
