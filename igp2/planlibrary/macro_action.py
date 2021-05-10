import abc
from typing import Dict, List
import numpy as np

from igp2.agent import AgentState
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, FollowLane, ManeuverConfig
from igp2.trajectory import VelocityTrajectory


class MacroAction(abc.ABC):
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True):
        self.open_loop = open_loop
        self.agent_id = agent_id
        self.start_frame = frame
        self.scenario_map = scenario_map
        self._maneuvers = None

    def done(self) -> bool:
        """ Returns True if the execution of the macro action has completed. """
        raise NotImplementedError

    def applicable(self, agent_id: int, frame: Dict[int, AgentState]) -> bool:
        """ Return True if the macro action is applicable in the given state of the environment. """
        raise NotImplementedError

    def get_trajectory(self) -> VelocityTrajectory:
        """ If open_loop is True then get the complete trajectory of the macro action.

        Returns:
            A VelocityTrajectory that describes the complete open loop trajectory of the macro action
        """
        if not self.open_loop:
            raise ValueError("Cannot get trajectory of closed-loop macro action!")
        if self._maneuvers is None:
            raise ValueError("Maneuver sequence of macro action was not initialised!")

        points = None
        velocity = None
        for maneuver in self._maneuvers:
            trajectory = maneuver.trajectory
            points = trajectory.path if points is None else np.append(points, trajectory.path, axis=0)
            velocity = trajectory.velocity if velocity is None else np.append(velocity, trajectory.velocity, axis=0)
        return VelocityTrajectory(points, velocity)

    @property
    def current_maneuver(self) -> Maneuver:
        """ The current maneuver being executed. """
        raise NotImplementedError

    @property
    def maneuvers(self):
        """ The complete maneuver sequence of the macro action. """
        return self._maneuvers


class Continue(MacroAction):
    def __init__(self, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map, open_loop: bool = True):
        super(Continue, self).__init__(agent_id, frame, scenario_map, open_loop)

        if self.open_loop:
            current_lane = scenario_map.best_lane_at(frame[agent_id].position, frame[agent_id].heading)
            endpoint = current_lane.midline.interpolate(1, normalized=True)
            config_dict = {
                "type": "open_loop",
                "termination_point": np.array(endpoint.coords[0])
            }
            config = ManeuverConfig(config_dict)
            self._maneuvers = [FollowLane(config, agent_id, frame, scenario_map)]

    def applicable(self, agent_id: int, frame: Dict[int, AgentState]) -> bool:
        return True
