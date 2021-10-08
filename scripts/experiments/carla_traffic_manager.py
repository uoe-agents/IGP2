import carla
import numpy as np

from igp2 import setup_logging
from igp2.agents.agentstate import AgentState
from igp2.agents.mcts_agent import MCTSAgent
from igp2.carla.carla_client import CarlaSim
from igp2.goal import PointGoal

setup_logging()

#carla_path = "C:\\Users\\Balint\\Documents\\Agents\\Carla"
carla_path = '/opt/carla-simulator'

scenario = "town01"
xodr_path = f"scenarios/maps/{scenario}.xodr"

frame = {
    0: AgentState(time=0,
                  position=np.array([92.21, -100.10]),
                  velocity=np.array([2.0, 0.0]),
                  acceleration=np.array([0.0, 0.0]),
                  heading=np.pi/2)
}

simulation = CarlaSim(xodr=xodr_path, carla_path=carla_path, rendering=True)

ego_id = 0
ego_goal = PointGoal(np.array((137.3, -59.43)), 1.5)
ego_agent = MCTSAgent(agent_id=ego_id, initial_state=frame[ego_id],
                      t_update=1.0, scenario_map=simulation.scenario_map, goal=ego_goal)
ego_actor = simulation.add_agent(ego_agent)
location = carla.Location(x=ego_agent.state.position[0], y=-ego_agent.state.position[1], z=50)
rotation = carla.Rotation(pitch=-70, yaw=-90)
transform = carla.Transform(location, rotation)
simulation.spectator.set_transform(transform)

tm = simulation.get_traffic_manager()
tm.set_agents_count(5)
tm.set_ego_agent(ego_agent)
tm.set_spawn_speed(low=4, high=14)

# get ego agent id
simulation.step()
ego_agent_actor_id = simulation.agents[ego_id].actor_id  # error: agents dict is empty
print(f'ego actor id: {ego_agent_actor_id}')

simulation.run(1000)
