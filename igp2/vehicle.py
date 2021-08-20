import numpy as np
from dataclasses import dataclass

from igp2.agentstate import AgentState, AgentMetadata
from igp2.util import Box


@dataclass(eq=True, frozen=True)
class Action:
    """ Action of acceleration and steering to execute by a vehicle. """
    acceleration: float
    steering: float


class Vehicle(Box):
    """ Class describing a physical vehicle object based on a bicycle-model. """
    def __init__(self, state: AgentState, meta: AgentMetadata):
        super().__init__(state.position, meta.length, meta.width, state.heading)
        self.velocity = state.speed
        self.acceleration = 0.0

        self._collision = None

    def execute_action(self, action: Action):
        pass  # TODO: Complete with bicycle model

    def get_state(self, time: float = None) -> AgentState:
        """ Return current state of the vehicle. """
        return AgentState(
            time=time,
            position=self.center,
            velocity=self.velocity,
            acceleration=self.acceleration,
            heading=self.heading
        )

    @property
    def collision(self) -> "Vehicle":

        return self._collision