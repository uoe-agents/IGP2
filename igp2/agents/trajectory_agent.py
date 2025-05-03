import numpy as np
import logging
from typing import Optional, Tuple, Union

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.core.vehicle import Action, Observation, TrajectoryVehicle, KinematicVehicle
from igp2.core.trajectory import StateTrajectory, VelocityTrajectory, Trajectory
from igp2.planlibrary.maneuver_cl import TrajectoryManeuverCL
from igp2.planlibrary.maneuver import ManeuverConfig

logger = logging.getLogger(__name__)


class TrajectoryAgent(Agent):
    """ Agent that follows a predefined trajectory. """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal: Goal = None,
                 fps: int = 20,
                 open_loop: bool = False,
                 reset_trajectory: bool = True):
        """ Initialise new trajectory-following agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            goal: Optional final goal of the vehicle
            fps: Execution rate of the environment simulation
            open_loop: Whether to use open-loop predictions directly instead of closed-loop control
            reset_trajectory: Whether to reset the trajectory of the agent when calling reset()
        """
        super().__init__(agent_id, initial_state, goal, fps)

        self._t = 0
        self._open_loop = open_loop
        self._reset_trajectory = reset_trajectory
        self._trajectory = None
        self._maneuver_config = None
        self._maneuver = None
        self._init_vehicle()

    def __repr__(self):
        if self.trajectory is not None:
            return f"TrajectoryAgent(ID={self.agent_id}, End={np.round(self.trajectory.path[-1], 2)})"
        else:
            return f"TrajectoryAgent(ID={self.agent_id})"

    def done(self, observation: Observation) -> bool:
        if self.open_loop:
            done = self._t == len(self._trajectory.path) - 1
        else:
            dist = np.linalg.norm(self._trajectory.path[-1] - observation.frame[self.agent_id].position)
            done = dist < 1.0  # arbitrary
        return done

    def next_action(self, observation: Observation) -> Optional[Action]:
        """ Calculate next action based on trajectory and optionally steps
        the current state of the agent forward. """
        assert self._trajectory is not None, f"Trajectory of Agent {self.agent_id} was None!"
        if self.done(observation):
            return None

        self._t += 1

        if self.open_loop:
            action = Action(self._trajectory.acceleration[self._t],
                            self._trajectory.angular_velocity[self._t])
        else:
            if self._maneuver is None:
                self._maneuver_config = ManeuverConfig({'type': 'trajectory',
                                                        'termination_point': self._trajectory.path[-1]})
                self._maneuver = TrajectoryManeuverCL(self._maneuver_config, self.agent_id, observation.frame,
                                                      observation.scenario_map, self._trajectory)
            action = self._maneuver.next_action(observation)

        return action

    def set_start_time(self, t: int):
        """ Set the current time step of the agent. """
        if t >= len(self._trajectory.path):
            logger.warning(f"Invalid time step {t} for Agent {self.agent_id}")
            t = len(self._trajectory.path) - 1
        self._t = t
        self._init_vehicle(self._get_open_loop_state())

    def next_state(self,
                   observation: Observation,
                   return_action: bool = False) \
            -> Union[AgentState, Tuple[AgentState, Action]]:
        """ Calculate next action based on trajectory, set appropriate fields in vehicle
        and returns the next agent state. """
        assert self._trajectory is not None, f"Trajectory of Agent {self.agent_id} was None!"
        if self.done(observation):
            return self.state

        action = self.next_action(observation)

        if self.open_loop:
            new_state = self._get_open_loop_state()
        else:
            new_state = None

        self.vehicle.execute_action(action, new_state)
        next_state = self.vehicle.get_state(self._t)

        if not return_action:
            return next_state
        else:
            return next_state, action

    def set_trajectory(self, new_trajectory: Trajectory, stop_seconds: float = 10., fps: int = None):
        """ Override current trajectory of the vehicle and resample to match execution frequency of the environment.
        If the trajectory given is empty or None, then the vehicle will stay in place for `stop_seconds` seconds. """
        fps = self._vehicle.fps if fps is None else fps
        if not new_trajectory:
            steps = int(stop_seconds * fps)
            self._trajectory = VelocityTrajectory(
                np.repeat([self._initial_state.position], steps, axis=0),
                np.zeros(steps),
                np.repeat(self._initial_state.heading, steps),
                np.repeat(1 / fps, steps)
            )

        elif isinstance(new_trajectory, StateTrajectory) and new_trajectory.fps == fps:
            self._trajectory = VelocityTrajectory(
                new_trajectory.path, new_trajectory.velocity,
                new_trajectory.heading, new_trajectory.timesteps)

        else:
            num_frames = np.ceil(new_trajectory.duration * fps)
            ts = new_trajectory.times
            points = np.linspace(ts[0], ts[-1], int(num_frames))

            xs_r = np.interp(points, ts, new_trajectory.path[:, 0])
            ys_r = np.interp(points, ts, new_trajectory.path[:, 1])
            v_r = np.interp(points, ts, new_trajectory.velocity)
            path = np.c_[xs_r, ys_r]
            self._trajectory = VelocityTrajectory(path, v_r)

    def reset(self):
        super(TrajectoryAgent, self).reset()
        self._t = 0
        if self._reset_trajectory:
            self._trajectory = None
        self._maneuver_config = None
        self._maneuver = None
        self._init_vehicle()

    def _init_vehicle(self, initial_state: AgentState = None):
        """ Create vehicle object of this agent. """
        if initial_state is None:
            initial_state = self._initial_state
        if self.open_loop:
            self._vehicle = TrajectoryVehicle(initial_state, self.metadata, self._fps)
        else:
            self._vehicle = KinematicVehicle(initial_state, self.metadata, self._fps)

    def _get_open_loop_state(self) -> AgentState:
        """ Returns the open-loop state at the current internal time step of the agent. """
        return AgentState(
            self._t,
            self._trajectory.path[self._t],
            self._trajectory.velocity[self._t],
            self._trajectory.acceleration[self._t],
            self._trajectory.heading[self._t]
        )

    @property
    def state(self):
        return self.vehicle.get_state(self._t)

    @property
    def trajectory(self) -> Trajectory:
        """ Return the currently defined trajectory of the agent. """
        return self._trajectory

    @property
    def open_loop(self) -> bool:
        """ Whether to use open-loop predictions directly instead of closed-loop control. """
        return self._open_loop

    def parked(self, tol=1.0) -> bool:
        return np.linalg.norm(self.trajectory.path[0] - self.trajectory.path[-1]) < tol
