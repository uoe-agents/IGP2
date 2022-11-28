import numpy as np

import igp2 as ip
import carla
from typing import Optional, List
from igp2.carla.local_planner import LocalPlanner, RoadOption
from igp2.carla.util import get_speed


class CarlaAgentWrapper:
    """ Wrapper class that provides a simple way to retrieve control for the attached actor. """

    def __init__(self, agent: ip.Agent, actor: carla.Actor):
        self.__agent = agent
        self.__actor = actor
        self.__name = self.__actor.attributes["role_name"]

        self.__world = self.__actor.get_world()
        self.__map = self.__world.get_map()

        self.__local_planner = LocalPlanner(self.__actor, self.__world, self.__map,
                                            dt=1.0 / self.__agent.fps)
        self.__waypoints = []  # List of CARLA waypoints to follow
        self.__current_ma = None

    def __repr__(self):
        return f"Actor {self.actor_id}; Agent {self.agent_id}"

    def next_control(self, observation: ip.Observation) -> Optional[carla.VehicleControl]:
        limited_observation = self.__apply_view_radius(observation)
        action = self.__agent.next_action(limited_observation)
        self.agent.vehicle.execute_action(action, observation.frame[self.agent_id])
        if action is None or self.__agent.done(observation):
            return None

        if hasattr(self.agent, "current_macro"):
            if self.__current_ma != self.agent.current_macro:
                self.__current_ma = self.agent.current_macro
                self.__trajectory_to_waypoints(self.__current_ma.get_trajectory())
                self.__local_planner.set_global_plan(
                    self.__waypoints, stop_waypoint_creation=True, clean_queue=True)

        target_speed = action.target_speed
        self.__local_planner.set_speed(target_speed * 3.6)
        return self.__local_planner.run_step()

    def done(self, observation: ip.Observation) -> bool:
        """ Returns whether the wrapped agent is done. """
        return self.__agent.done(observation)

    def reset_waypoints(self):
        self.__waypoints = []

    def __apply_view_radius(self, observation: ip.Observation):
        if hasattr(self.agent, "view_radius"):
            pos = observation.frame[self.agent_id].position
            new_frame = {aid: state for aid, state in observation.frame.items()
                         if np.linalg.norm(pos - state.position) <= self.agent.view_radius}
            return ip.Observation(new_frame, observation.scenario_map)
        return observation

    def __trajectory_to_waypoints(self, trajectory: ip.VelocityTrajectory):
        self.__waypoints = []
        for point in trajectory.path[:-1]:
            wp = self.__map.get_waypoint(carla.Location(point[0], -point[1]))
            wp = (wp, RoadOption.LANEFOLLOW)
            assert wp is not None, f"Invalid waypoint found at {point}."
            self.__waypoints.append(wp)

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
