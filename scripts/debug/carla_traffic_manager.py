import numpy as np

from igp2.agent.agentstate import AgentState
from igp2.agent.mcts_agent import MCTSAgent
from igp2.carla.carla_client import CarlaSim
from igp2.carla.traffic_manager import TrafficManager


carla_path = "C:\\Users\\Balint\\Documents\\Agents\\Carla"

scenario = "town01"
xodr_path = f"scenarios/maps/{scenario}.xodr"

frame = {
    0: AgentState(time=0,
                  position=np.array([92.28, -57.30]),
                  velocity=np.array([5.0, 0.0]),
                  acceleration=np.array([0.0, 0.0]),
                  heading=np.pi/2)
}

simulation = CarlaSim(xodr=xodr_path, carla_path=carla_path)

ego_id = 0
ego_agent = MCTSAgent(agent_id=ego_id, initial_state=frame[ego_id],
                      t_update=1.0, metadata=frame[ego_id].metadata,
                      scenario_map=simulation.scenario_map)
simulation.add_agent(ego_agent)

tm = simulation.traffic_manager
tm.set_agents_count(10)
tm.set_ego_agent(ego_agent)
tm.set_spawn_speed(low=4, high=14)

simulation.run()
