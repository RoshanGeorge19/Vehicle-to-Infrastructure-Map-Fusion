import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd

# df_odo = pd.read_csv("/data/lidar_data/CAR/Scenario-2/slam/trajectory_with_interpolated_gps.csv")
df_odo = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/slam/v1/trajectory_v1.csv")
df_ann = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/id-307_id-356_static_sign_2_session_converted_car_final.csv")

# Round the 'Time' column in both dataframes to ensure proper matching
# df_ann['Time_Short'] = df_ann['Time_Short'].round(2)
# df_odo['Time'] = df_odo['Time'] - 0.05
# df_odo['Time'] = df_odo['Time'].round(2)
df_merged = pd.merge_asof(df_ann, df_odo, left_on='Time_Short', right_on='Time', direction='nearest')

# Initialize lists to store pedestrian world coordinates
pedestrian_world_coords = []


for index, row in df_merged.iterrows():
    if row['X_Center'] != 0:
        pedestrian_relative_pos = np.array([row['X_Center'], row['Y_Center'], row['Z_Center']])
        pedestrian_world_pos = pedestrian_relative_pos + np.array([row['X'], row['Y'], row['Z']])
        # pedestrian_world_pos = pedestrian_relative_pos + np.array([0, 0, 0])
        pedestrian_world_coords.append(pedestrian_world_pos)

pedestrian_world_coords = np.array(pedestrian_world_coords)

# Plotting the trajectory of the pedestrian
plt.figure(figsize=(10, 6))
plt.plot(pedestrian_world_coords[:, 0], pedestrian_world_coords[:, 1], marker='o', linestyle='-', color='b')
plt.title("Pedestrian Trajectory in World Coordinate System")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()
