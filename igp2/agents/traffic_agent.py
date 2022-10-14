from typing import List

import igp2 as ip
from igp2.agents.macro_agent import MacroAgent
import logging

logger = logging.getLogger(__name__)


class TrafficAgent(MacroAgent):
    """ Agent that follows a list of MAs, optionally calculated using A*. """

    def __init__(self, agent_id: int, initial_state: ip.AgentState, goal: "ip.Goal" = None, fps: int = 20):
        super(TrafficAgent, self).__init__(agent_id, initial_state, goal, fps)
        self._astar = ip.AStar(max_iter=1000)
        self._macro_actions = []

    def set_macro_actions(self, new_macros: List[ip.MacroAction]):
        """ Specify a new set of macro actions to follow. """
        assert len(new_macros) > 0, "Empty macro list given!"
        self._macro_actions = new_macros

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

    def done(self, observation: ip.Observation) -> bool:
        """ Returns true if there are no more actions on the macro list and the current macro is finished. """
        return len(self._macro_actions) == 0 and super(TrafficAgent, self).done(observation)

    def next_action(self, observation: ip.Observation) -> ip.Action:
        if self.current_macro is None:
            if len(self._macro_actions) == 0:
                self.set_destination(observation)
            self._advance_macro(observation)

        if self._current_macro.done(observation):
            if len(self._macro_actions) > 0:
                self._advance_macro(observation)
            else:
                return ip.Action(0, 0)

        return self._current_macro.next_action(observation)

    def _advance_macro(self, observation: ip.Observation):
        self._current_macro = self._macro_actions.pop(0)

    @property
    def macro_actions(self) -> List[ip.MacroAction]:
        """ The current macro actions to be executed by the agent. """
        return self._macro_actions
