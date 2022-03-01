import igp2 as ip
import logging

logger = logging.getLogger(__name__)


class TrafficAgent(ip.MacroAgent):
    """ Agent that follows a list of MAs calculated using A*. """

    def __init__(self, agent_id: int, initial_state: ip.AgentState, goal: "ip.Goal" = None, fps: int = 20):
        super(TrafficAgent, self).__init__(agent_id, initial_state, goal, fps)
        self._astar = ip.AStar(max_iter=1000)
        self._macro_list = []

    def set_destination(self, goal: ip.Goal, scenario_map: ip.Map):
        """ Set the current destination of this vehicle and calculate the shortest path to it using A*. """
        logger.debug(f"Finding path for TrafficAgent ID {self.agent_id}")
        self._goal = goal
        _, actions = self._astar.search(self.agent_id, {self.agent_id: self._vehicle.get_state()},
                                        goal, scenario_map, open_loop=False)
        self._macro_list = actions[0]

    def done(self, observation: ip.Observation) -> bool:
        """ Returns true if there are no more actions on the macro list and the current macro is finished. """
        return len(self._macro_list) == 0 and super(TrafficAgent, self).done(observation)

    def next_action(self, observation: ip.Observation) -> ip.Action:
        if self.current_macro is None:
            if len(self._macro_list) > 0:
                self._advance_macro()
            else:
                return ip.Action(0, 0)

        if self._current_macro.done(observation):
            if len(self._macro_list) > 0:
                self._advance_macro()
            else:
                return ip.Action(0, 0)

        return self._current_macro.next_action(observation)

    def next_state(self, observation: ip.Observation) -> ip.AgentState:
        return super(TrafficAgent, self).next_state(observation)

    def _advance_macro(self):
        self._current_macro = self._macro_list.pop(0)
