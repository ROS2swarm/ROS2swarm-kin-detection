import pandas as pd
from experiment_measurement.rosbag2df import read_rosbag_all_in_one, Rosbag2Df
from experiment_measurement.data_aggregation import aggregate_tables
import rclpy 
### change the following line according to what config you want to use
from experiment_measurement.config.lidar_data import table_column_config

# read data from sqlite db
data = read_rosbag_all_in_one('rosbag/rosbag_4_.db3')

# export one dataframe per robot 
for robot in tables.keys():
    tables[robot].dropna(subset=['scan', 'poses']).to_csv(path_or_buf='rosbag_'+str(robot)+'.csv', index=False)
    
