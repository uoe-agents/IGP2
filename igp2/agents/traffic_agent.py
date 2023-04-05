from typing import List

import igp2 as ip
from igp2.agents.macro_agent import MacroAgent
import logging

logger = logging.getLogger(__name__)


class TrafficAgent(MacroAgent):
    """ Agent that follows a list of MAs, optionally calculated using A*. """

    def __init__(self, agent_id: int, initial_state: ip.AgentState, goal: "ip.Goal" = None, fps: int = 20,
                 macro_actions: List[ip.MacroAction] = None):
        super(TrafficAgent, self).__init__(agent_id, initial_state, goal, fps)
        self._astar = ip.AStar(max_iter=1000)
        self._macro_actions = []
        if macro_actions is not None:
            self.set_macro_actions(macro_actions)
        self._current_macro_id = 0

    def set_macro_actions(self, new_macros: List[ip.MacroAction]):
        """ Specify a new set of macro actions to follow. """
        assert len(new_macros) > 0, "Empty macro list given!"
        for macro in new_macros:
            macro.to_closed_loop()
        self._macro_actions = new_macros
        self._current_macro = new_macros[0]

    def set_destination(self, observation: ip.Observation, goal: ip.Goal = None):
        """ Set the current destination of this vehicle and calculate the shortest path to it using A*.

            Args:
                observation: The current observation.
                goal: Optional new goal to override the current one.
        """
        if goal is not None:
            self._goal = goal

        logger.debug(f"Finding path for TrafficAgent ID {self.agent_id}")
        _, actions = self._astar.search(self.agent_id,
                                        observation.frame,
                                        self._goal,
                                        observation.scenario_map,
                                        open_loop=False)

        if len(actions) == 0:
            raise RuntimeError(f"Couldn't find path to goal {self.goal} for TrafficAgent {self.agent_id}.")
        self._macro_actions = actions[0]
        self._current_macro = self._macro_actions[0]

    def done(self, observation: ip.Observation) -> bool:
        """ Returns true if there are no more actions on the macro list and the current macro is finished. """
        return self._current_macro_id + 1 >= len(self._macro_actions) and super(TrafficAgent, self).done(observation)

    def next_action(self, observation: ip.Observation) -> ip.Action:
        if self.current_macro is None:
            if len(self._macro_actions) == 0:
                self.set_destination(observation)

        if self._current_macro.done(observation):
            if self._current_macro_id < len(self._macro_actions):
                self._advance_macro(observation)
            else:
                logger.warning(f"TrafficAgent {self.agent_id} has no macro actions!")
                return ip.Action(0, 0)

        return self._current_macro.next_action(observation)

    def reset(self):
        super(TrafficAgent, self).reset()
        self._macro_actions = []
        self._current_macro_id = 0

    def _advance_macro(self, observation: ip.Observation):
        if not self._macro_actions:
            raise RuntimeError("TrafficAgent has no macro actions.")

        self._current_macro_id += 1
        if self._current_macro_id >= len(self._macro_actions):
            raise RuntimeError("No more macro actions to execute.")
        self._current_macro = self._macro_actions[self._current_macro_id]

    @property
    def macro_actions(self) -> List[ip.MacroAction]:
        """ The current macro actions to be executed by the agent. """
        return self._macro_actions
