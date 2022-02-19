import igp2 as ip
import matplotlib.pyplot as plt

scenario_path = "scenarios/maps/test_lane_offsets.xodr"
scenario_map = ip.Map.parse_from_opendrive(scenario_path)
ip.plot_map(scenario_map, markings=True, midline=True)
plt.show()