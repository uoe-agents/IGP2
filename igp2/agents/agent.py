import abc
from typing import Optional

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.trajectory import StateTrajectory
from igp2.core.vehicle import Observation, Action
from igp2.core.util import Box


class Agent(abc.ABC):
    """ Abstract class for all agents. """

    def __init__(self, agent_id: int, initial_state: AgentState, goal: "Goal" = None, fps: int = 20):
        """ Initialise base fields of the agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            goal: Optional final goal of the agent
            fps: Execution rate of the environment simulation
        """
        self._alive = True

        self._agent_id = agent_id
        self._initial_state = initial_state
        self._goal = goal
        self._fps = fps
        self._vehicle = None
        self._trajectory_cl = StateTrajectory(self._fps)
        self._trajectory_cl.add_state(self._initial_state)

    def done(self, observation: Observation) -> bool:
        """ Check whether the agent has completed executing its assigned task. """
        raise NotImplementedError

    def next_action(self, observation: Observation) -> Action:
        """ Return the next action the agent will take"""
        raise NotImplementedError

    def next_state(self, observation: Observation, return_action: bool = False) -> AgentState:
        """ Return the next agent state after it executes an action. """
        raise NotImplementedError

    def update_goal(self, new_goal: "Goal"):
        """ Overwrite the current goal of the agent. """
        self._goal = new_goal

    def reset(self):
        """ Reset agent to initialisation defaults. """
        self._alive = True
        self._vehicle = None
        self._trajectory_cl = StateTrajectory(self._fps)
        self._trajectory_cl.add_state(self._initial_state)

    @property
    def agent_id(self) -> int:
        """ ID of the agent. """
        return self._agent_id

    @property
    def state(self) -> AgentState:
        """ Return current state of the agent as given by its vehicle, or initial state if no vehicle is attached. """
        if self._vehicle is not None:
            return self._vehicle.get_state()
        else:
            return self._initial_state

    @property
    def metadata(self) -> AgentMetadata:
        """ Metadata describing the physical properties of the agent. """
        return self._initial_state.metadata

    @property
    def goal(self) -> "Goal":
        """ Final goal of the agent. """
        return self._goal

    @property
    def vehicle(self) -> "Vehicle":
        """ Return the physical vehicle attached to this agent. """
        return self._vehicle

    @property
    def alive(self) -> bool:
        """ Whether the agent is alive in the simulation. """
        return self._alive

    @alive.setter
    def alive(self, value: bool):
        self._alive = value

    @property
    def trajectory_cl(self):
        """ The closed loop trajectory that was actually driven by the agent. """
        return self._trajectory_cl

    @property
    def fps(self) -> int:
        """ Simulation frames per second. """
        return self._fps

    @property
    def bounding_box(self) -> Optional[Box]:
        """ Return the bounding box of the agent. """
        state = self.state
        if state is not None:
            return Box(self.state.position, self.metadata.length, self.metadata.width, self.state.heading)
        return None
