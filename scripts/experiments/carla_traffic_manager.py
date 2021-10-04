import carla
import numpy as np

from igp2 import setup_logging
from igp2.agents.agentstate import AgentState
from igp2.agents.mcts_agent import MCTSAgent
from igp2.carla.carla_client import CarlaSim

setup_logging()

carla_path = "C:\\Users\\Balint\\Documents\\Agents\\Carla"

scenario = "town01"
xodr_path = f"scenarios/maps/{scenario}.xodr"

frame = {
    0: AgentState(time=0,
                  position=np.array([73.2, -47.1]),
                  velocity=np.array([-5.0, 0.5]),
                  acceleration=np.array([0.0, 0.0]),
                  heading=np.pi + np.pi/3)
}

simulation = CarlaSim(xodr=xodr_path, carla_path=carla_path)

ego_id = 0
ego_agent = MCTSAgent(agent_id=ego_id, initial_state=frame[ego_id],
                      t_update=1.0, scenario_map=simulation.scenario_map)
# ego_actor = simulation.add_agent(ego_agent)
# location = carla.Location(x=ego_agent.state.position[0], y=ego_agent.state.position[1], z=100)
# rotation = carla.Rotation(pitch=-70)
# transform = carla.Transform(location, rotation)
# simulation.spectator.set_transform(transform)

tm = simulation.get_traffic_manager()
tm.set_agents_count(5)
tm.set_ego_agent(ego_agent)
tm.set_spawn_speed(low=4, high=14)

simulation.run(1000)
