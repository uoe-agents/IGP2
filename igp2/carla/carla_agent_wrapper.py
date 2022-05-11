import numpy as np

import igp2 as ip
import carla
from typing import Optional


class CarlaAgentWrapper:
    MAX_ACCELERATION = 5

    """ Wrapper class that provides a simple way to retrieve control for the attached actor. """
    def __init__(self, agent: ip.Agent, actor: carla.Actor):
        self.__agent = agent
        self.__actor = actor
        self.__name = self.__actor.attributes["role_name"]

    def __repr__(self):
        return f"Actor {self.actor_id}; Agent {self.agent_id}"

    def next_control(self, observation: ip.Observation) -> Optional[carla.VehicleControl]:
        limited_observation = self._apply_view_radius(observation)
        action = self.__agent.next_action(limited_observation)
        if action is None or self.__agent.done(observation):
            return None

        control = carla.VehicleControl()
        norm_acceleration = action.acceleration / (0.95 * self.__agent.fps)  # self.MAX_ACCELERATION
        if action.acceleration >= 0:
            control.throttle = min(1., norm_acceleration)
            control.brake = 0.
        else:
            control.throttle = 0.
            control.brake = min(-norm_acceleration, 1.)
        control.steer = -action.steer_angle
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def done(self, observation: ip.Observation) -> bool:
        """ Returns whether the wrapped agant is done. """
        return self.__agent.done(observation)

    def _apply_view_radius(self, observation: ip.Observation):
        if hasattr(self.agent, "view_radius"):
            pos = observation.frame[self.agent_id].position
            new_frame = {aid: state for aid, state in observation.frame.items()
                         if np.linalg.norm(pos - state.position) <= self.agent.view_radius}
            return ip.Observation(new_frame, observation.scenario_map)
        return observation

    @property
    def state(self) -> ip.AgentState:
        return self.agent.state

    @property
    def agent_id(self) -> int:
        return self.__agent.agent_id

    @property
    def actor_id(self) -> int:
        return self.__actor.id

    @property
    def actor(self) -> carla.Actor:
        return self.__actor

    @property
    def agent(self) -> ip.Agent:
        return self.__agent

    @property
    def name(self):
        """ The role name of the wrapped Actor. """
        return self.__name
