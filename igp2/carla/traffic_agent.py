from igp2.agents.agentstate import AgentState
from igp2.agents.macro_agent import MacroAgent
from igp2.goal import PointGoal
from igp2.opendrive.map import Map
from igp2.recognition.astar import AStar
from igp2.vehicle import Observation, Action, KinematicVehicle


class TrafficAgent(MacroAgent):
    """ Agent that follows a list of MAs calculated using A*. """

    def __init__(self, agent_id: int, initial_state: AgentState, goal: "Goal" = None, fps: int = 20):
        super(TrafficAgent, self).__init__(agent_id, initial_state, goal, fps)
        self._astar = AStar(max_iter=1000)
        self._macro_list = []
        if goal is not None:
            self.set_destination(goal)

    def set_destination(self, goal: PointGoal, scenario_map: Map):
        """ Set the current destination of this vehicle and calculate the shortest path to it using A*. """
        self._goal = goal
        _, actions = self._astar.search(self.agent_id, {self.agent_id: self._vehicle.get_state()},
                                        goal, scenario_map, open_loop=False)
        self._macro_list = actions[0]

    def done(self, observation: Observation) -> bool:
        """ Returns true if there are no more actions on the macro list and the current macro is finished. """
        return len(self._macro_list) == 0 and super(TrafficAgent, self).done(observation)

    def next_action(self, observation: Observation) -> Action:
        pass

    def next_state(self, observation: Observation) -> AgentState:
        pass
