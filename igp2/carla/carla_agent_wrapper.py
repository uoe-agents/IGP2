import igp2 as ip
import carla
from typing import Optional


class CarlaAgentWrapper:
    MAX_ACCELERATION = 5

    """ Wrapper class that provides a simple way to retrieve control for the attached actor. """
    def __init__(self, agent: ip.Agent, actor: carla.Actor):
        self.__agent = agent
        self.__actor = actor

    def next_control(self, observation: ip.Observation) -> Optional[carla.VehicleControl]:
        action = self.__agent.next_action(observation)
        if action is None or self.__agent.done(observation):
            return None

        control = carla.VehicleControl()
        norm_acceleration = action.acceleration / self.MAX_ACCELERATION
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
