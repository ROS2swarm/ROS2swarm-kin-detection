import pandas as pd
from experiment_measurement.rosbag2df import read_rosbag_all_in_one, Rosbag2Df
from experiment_measurement.data_aggregation import aggregate_tables
import rclpy 
import os 
import glob
### change the following line according to what config you want to use
from experiment_measurement.config.lidar_data import table_column_config



# get rosbag folders, exclude hidden folders 
subfolders = [f.name for f in os.scandir('./') if f.is_dir() and f.name[:1] != '.']

# read data from sqlite db
for subfolder in subfolders: 
    files = glob.glob(subfolder+'/*.db3')
    for f in files: 
        # read in file 
        data = read_rosbag_all_in_one(f)
        
        # create pandas dataframe 
        tables = aggregate_tables(data['rosbag'], table_column_config, 10**9)
        
        # export one dataframe per robot 
        for robot in tables.keys():
            tables[robot].dropna(subset=['scan', 'poses']).to_csv(path_or_buf=str(subfolder) + '/' + str(robot)+'.csv', index=False)
    
    





    
