import igp2 as ip
import matplotlib.pyplot as plt

scenario_path = "scenarios/maps/scenario2.xodr"
scenario_map = ip.Map.parse_from_opendrive(scenario_path)
ip.plot_map(scenario_map, markings=False, midline=True)
plt.show()