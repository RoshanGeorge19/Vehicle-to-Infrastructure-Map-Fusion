import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory_car(df):
    plt.figure()
    plt.scatter(df['X'], df['Y'], c=df['Time'], cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-50, 50)
    plt.grid(True)
    plt.colorbar(label='Time')
    plt.show()

def verify_object_relative_to_lidar_car(df_annotations, df_odo):
    # Round the 'Time' column in both dataframes to ensure proper matching
    df_annotations['Time_Short'] = df_annotations['Time_Short'].round(2)
    df_odo['Time'] = df_odo['Time'] - 0.05
    df_odo['Time'] = df_odo['Time'].round(2)

    df_odo['Shifted_Time'] = df_odo['Time'].shift(-4)
    df3 = pd.merge(df_annotations, df_odo, left_on='Time_Short', right_on='Shifted_Time', how='left')
    df_merged = pd.merge_asof(df_annotations, df_odo, left_on='Time_Short', right_on='Time', direction='nearest')

    pedestrian_world_coords = []
    theta = np.deg2rad(-15)
    R_LidarToLocal = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]



    for index, row in df_merged.iterrows():
        if row['X_Center'] != 0:
            Rx = row['Rx(Roll)']
            Ry = row['Ry(Pitch)']
            Rz = row['Rz(Yaw)']

            R_x = np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            R_y = np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
            R_z = np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            R_Odom = R_z @ R_y @ R_x

            pedestrian_relative_pos_lidar_cs = np.array([row['X_Center'], row['Y_Center'], row['Z_Center']])
            pedestrian_relative_pos_local_cs = np.dot(R_LidarToLocal, pedestrian_relative_pos_lidar_cs)

            pedestrian_pos_rel_to_sensor = pedestrian_relative_pos_local_cs + np.array([row['X'], row['Y'], row['Z']])
            # pedestrian_pos_rel_to_sensor = pedestrian_relative_pos_local_cs + np.dot(R_Odom, np.array([row['X'], row['Y'], row['Z']]))
            pedestrian_world_coords.append(pedestrian_pos_rel_to_sensor)

    pedestrian_world_coords = np.array(pedestrian_world_coords)

    plt.figure()
    plt.scatter(pedestrian_world_coords[:, 0], pedestrian_world_coords[:, 1], c=range(len(pedestrian_world_coords)), cmap='viridis')
    # plt.scatter(pedestrian_world_coords[0:5, 0], pedestrian_world_coords[0:5, 1])
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.grid(True)
    plt.axis('equal')
    plt.colorbar(label='Index')
    plt.show()

def verify_object_relative_to_lidar_node(df_annotations):
    pedestrian_world_coords = []

    for index, row in df_annotations.iterrows():
        if row['X_Center'] != 0:
            pedestrian_relative_pos = np.array([row['X_Center'], row['Y_Center'], row['Z_Center']])
            pedestrian_world_pos = pedestrian_relative_pos + np.array([0, 0, 0])
            pedestrian_world_coords.append(pedestrian_world_pos)

    pedestrian_world_coords = np.array(pedestrian_world_coords)

    plt.figure()
    plt.scatter(pedestrian_world_coords[:, 0], pedestrian_world_coords[:, 1], c=range(len(pedestrian_world_coords)), cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.grid(True)
    plt.axis('equal')
    plt.colorbar(label='Index')
    plt.show()




def main():
    df_annotations_car = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_3_annotations_car.csv")
    df_annotations_node = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_3_annnotations_node.csv")

    df_lidar_cs = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/v2/trajectory_v2.csv')
    df_local_cs = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/v2/trajectory_v2_local_cs.csv')
    df_global_cs = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/v2/trajectory_v2_global_cs.csv')

    plot_trajectory_car(df_lidar_cs)
    plot_trajectory_car(df_local_cs)
    plot_trajectory_car(df_global_cs)

    df_odo = df_local_cs
    verify_object_relative_to_lidar_car(df_annotations_car, df_odo)
    verify_object_relative_to_lidar_node(df_annotations_node)

    print(df_annotations_car.head())
    print(df_annotations_node.head())


if __name__ == '__main__':
    main()