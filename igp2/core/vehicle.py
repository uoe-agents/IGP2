from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.util import Box
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map


@dataclass(eq=True, frozen=True)
class Action:
    """ Represents an action taken by an agent"""
    acceleration: float
    steer_angle: float
    target_speed: float = None
    target_angle: float = None


@dataclass(eq=True, frozen=True)
class Observation:
    """ Represents an observation of the visible environment state and the road layout"""
    frame: Dict[int, AgentState]
    scenario_map: Map

    def plot(self, ax: plt.Axes = None) -> plt.Axes:
        """ Convenience method to plot the current observation. """
        if ax is None:
            fig, ax = plt.subplots()
        plot_map(self.scenario_map, ax, markings=True, midline=True)
        for aid, state in self.frame.items():
            ax.plot(*state.position, marker="o")
        return ax


class Vehicle(Box):
    """ Base class for physical vehicle control. """
    def __init__(self, state: AgentState, meta: AgentMetadata, fps: int):
        """ Initialise new vehicle.

        Args:
            state: Starting state of the vehicle
            meta: Metadata giving the physical properties of the vehicle
            fps: Execution frequency of the environment simulation
        """
        super().__init__(state.position, meta.length, meta.width, state.heading)
        self.velocity = state.speed
        self.acceleration = 0.0
        self.meta = meta
        self.fps = fps
        self._dt = 1 / fps

    def execute_action(self, action: Action = None, next_state: AgentState = None):
        """ Execute action given to the vehicle.

        Args:
            action: Acceleration and steering action to execute
            next_state: Can be used to override action and set state directly
        """
        raise NotImplementedError

    def get_state(self, time: float = None) -> AgentState:
        """ Return current state of the vehicle. """
        return AgentState(
            time=time,
            position=self.center.copy(),
            velocity=self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)]),
            acceleration=self.acceleration * np.array([np.cos(self.heading), np.sin(self.heading)]),
            heading=self.heading
        )


class TrajectoryVehicle(Vehicle):
    def execute_action(self, action: Action = None, next_state: AgentState = None):
        """ Used next_state to set the state of the vehicle manually as given by an already calculated trajectory.

        Args:
            action: Ignored
            next_state: Next state of the vehicle
        """
        if next_state is None:
            raise ValueError("No state given to TrajectoryVehicle.")

        self.center = next_state.position
        self.velocity = next_state.speed
        self.heading = next_state.heading
        self.acceleration = next_state.acceleration
        self.calculate_boundary()


class KinematicVehicle(Vehicle):
    """ Class describing a physical vehicle object based on a bicycle-model. """
    def __init__(self, state: AgentState, meta: AgentMetadata, fps: int):
        super().__init__(state, meta, fps)

        correction = (self.meta.rear_overhang - self.meta.front_overhang) / 2  # Correction for cg
        self._l_f = self.meta.wheelbase / 2 + correction  # Distance of front axel from cg
        self._l_r = self.meta.wheelbase / 2 - correction  # Distance of back axel from cg

    def execute_action(self, action: Action = None, next_state: AgentState = None) -> Action:
        """ Apply acceleration and steering according to the bicycle model centered at the
        center-of-gravity (i.e. cg) of the vehicle.

        Ref: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357

        Args:
            action: Acceleration and steering action to execute
            next_state: Ignored

        Returns:
            Acceleration and heading action that was executed by the vehicle.
        """
        self.acceleration = np.clip(action.acceleration, - self.meta.max_acceleration, self.meta.max_acceleration)
        self.velocity += self.acceleration * self._dt
        self.velocity = max(0, self.velocity)
        beta = np.arctan(self._l_r * np.tan(action.steer_angle) / self.meta.wheelbase)
        d_position = np.array(
            [self.velocity * np.cos(beta + self.heading),
             self.velocity * np.sin(beta + self.heading)]
        )
        self.center += d_position * self._dt
        d_theta = self.velocity * np.tan(action.steer_angle) * np.cos(beta) / self.meta.wheelbase
        d_theta = np.clip(d_theta, - self.meta.max_angular_vel, self.meta.max_angular_vel)
        self.heading = (self.heading + d_theta * self._dt + np.pi) % (2*np.pi) - np.pi

        # # Unicycle model
        # self.acceleration = np.clip(action.acceleration, - self.meta.max_acceleration, self.meta.max_acceleration)
        # self.velocity += self.acceleration * self._dt
        # self.velocity = max(0, self.velocity)
        # d_theta = action.steer_angle
        # #update heading but respect the (-pi, pi) convention
        # self.heading = (self.heading + d_theta * self._dt + np.pi) % (2*np.pi) - np.pi
        # d_position = np.array([self.velocity * np.cos(self.heading), self.velocity * np.sin(self.heading)])
        # self.center += d_position * self._dt

        self.calculate_boundary()
        return Action(float(self.acceleration), float(d_theta))
