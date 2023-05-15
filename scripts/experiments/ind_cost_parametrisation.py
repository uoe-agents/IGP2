import csv
import os

#This script extracts the limit dictionary used in the cost class 
# from the data/limit_values.csv file


def load_limit_values():
    filename = os.path.dirname(os.path.abspath(__file__))  + '/data/limit_values.csv'
    csv_file = open(filename, "r")
    dict_reader = csv.DictReader(csv_file)
    ordered_dict_from_csv = list(dict_reader)[0]
    limit_values = dict(ordered_dict_from_csv)
    for k, v in limit_values.items():
        limit_values[k] = float(v)
    return limit_values

all_limits = load_limit_values()

limits = {"acceleration": 0., "jerk": 0., "angular_velocity": 0.,
                         "angular_acceleration": 0., "curvature": 0.}

for key in limits.keys():
    minkey = "min_" + key
    maxkey = "max_" + key
    limits[key] = max(abs(all_limits[minkey]),abs(all_limits[maxkey]))