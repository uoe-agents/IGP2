import igp2 as ip
import numpy as np
from typing import List, Dict

from igp2.agents.agent import Agent


class MacroAgent(Agent):
    """ Agent executing a pre-defined macro action. Useful for simulating the ego vehicle during MCTS. """

    def __init__(self,
                 agent_id: int,
                 initial_state: ip.AgentState,
                 goal: ip.Goal = None,
                 fps: int = 20):
        """ Create a new macro agent. """
        super().__init__(agent_id, initial_state, goal, fps)
        self._vehicle = ip.KinematicVehicle(initial_state, self.metadata, fps)
        self._current_macro = None
        self._maneuver_end_idx = []

    @property
    def current_macro(self) -> ip.MacroAction:
        """ The current macro action of the agent. """
        return self._current_macro

    @property
    def maneuver_end_idx(self) -> List[int]:
        """ The closed loop trajectory id at which each macro action maneuver completes."""
        return self._maneuver_end_idx

    def done(self, observation: ip.Observation) -> bool:
        """ Returns true if the current macro action has reached a completion state. """
        assert self._current_macro is not None, f"Macro action of Agent {self.agent_id} is None!"
        return self._current_macro.done(observation)

    def next_action(self, observation: ip.Observation) -> ip.Action:
        """ Get the next action from the macro action.

        Args:
            observation: Observation of current environment state and road layout.

        Returns:
            The next action of the agent.
        """

        assert self._current_macro is not None, f"Macro action of Agent {self.agent_id} is None!"

        if self._current_macro.current_maneuver is not None and self._current_macro.current_maneuver.done(observation):
            self._maneuver_end_idx.append(len(self.trajectory_cl.states) - 1)
        return self._current_macro.next_action(observation)

    def next_state(self, observation: ip.Observation, return_action: bool = False) -> ip.AgentState:
        """ Get the next action from the macro action and execute it through the attached vehicle of the agent.

        Args:
            observation: Observation of current environment state and road layout.
            return_action: If True return the underlying action as well.

        Returns:
            The new state of the agent.
        """
        action = self.next_action(observation)
        self.vehicle.execute_action(action, observation.frame[self.agent_id])
        next_state = self.vehicle.get_state(observation.frame[self.agent_id].time + 1)
        next_state.macro_action = str(self.current_macro)
        next_state.maneuver = str(self.current_macro.current_maneuver)

        if not return_action:
            return next_state
        else:
            return next_state, action

    def reset(self):
        """ Reset the vehicle and macro action of the agent."""
        super(MacroAgent, self).reset()
        self._vehicle = ip.KinematicVehicle(self._initial_state, self.metadata, self._fps)
        self._current_macro = None

    def update_macro_action(self,
                            macro_action: type(ip.MacroAction),
                            args: Dict,
                            observation: ip.Observation) -> ip.MacroAction:
        """ Overwrite and initialise current macro action of the agent using the given arguments.

        Args:
            macro_action: new macro action to execute
            args: MA initialisation arguments
            observation: Observation of the environment
        """
        args["open_loop"] = False
        config = ip.MacroActionConfig(args)
        self._current_macro = macro_action(config,
                                           agent_id=self.agent_id,
                                           frame=observation.frame,
                                           scenario_map=observation.scenario_map)
        return self._current_macro
