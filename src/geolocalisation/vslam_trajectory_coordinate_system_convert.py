# This takes in a csv of the LiDAR SLAM odometry and outputs 1. trajectory csv with duplicates removed, 2. the trajectory roated to the local coordinate system, 3. the trajectory in the global coordinate system.
import pandas as pd
import numpy as np
from wp2.geo_utils import GeoTransformer

def get_slam_rotated_trajectory(df):
    # Copy the dataframe to avoid modifying the original dataframe
    df_rotated = df.copy()

    # for index, row in df.iterrows():
    #     Rx = row['Rx(Roll)']
    #     Ry = row['Ry(Pitch)']
    #     Rz = row['Rz(Yaw)']
    #
    #     R_x = np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
    #     R_y = np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
    #     R_z = np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
    #     R_Odom = R_z @ R_y @ R_x
    #
    #     df_rotated.at[index, ['X', 'Y', 'Z']] = np.dot(R_Odom, row[['X', 'Y', 'Z']].values)


    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0], [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0], [0, 0, 1]]
    df_rotated[['X', 'Y', 'Z']] = df_rotated.apply(lambda row: pd.Series(np.dot(R_lidar_to_local, row[['X', 'Y', 'Z']].values)), axis=1)
    return df_rotated

def convert_slam_rotated_trajectory_to_gps(GPS0, GPSX, GPSY, GPS_start, df_rotated, df_car_base, df, geo_transformer):
    # Use GPS0, GPSX, GPSY to get rotation matrix. LiDAR to Local.
    # Use GPS_start to get base GPS to offset.
    ECEF0 = geo_transformer.gps_to_ecef(*GPS0)
    ECEFX = geo_transformer.gps_to_ecef(*GPSX)
    ECEFY = geo_transformer.gps_to_ecef(*GPSY)
    R_local_to_global = geo_transformer.get_rotation(ECEF0, ECEFX, ECEFY)

    ECEF_start = geo_transformer.gps_to_ecef(*GPS_start)

    df_gps_list = []
    df_traj_gps_list = []

    for index, row in df_rotated.iterrows():
        lidar_point = np.array([-row['X'], row['Y'], row['Z']])
        # ecef_point = lidar_to_ecef(lidar_point, ECEF0, R_local_to_global)
        ecef_point = geo_transformer.lidar_to_ecef(lidar_point, ECEF_start, R_local_to_global)
        gps_point = geo_transformer.ecef_to_gps(*ecef_point)

        # Visualisation
        df_gps_list.append(pd.DataFrame({'Colour_ID': [0], 'Name': [row['Time']], 'Longitude': [gps_point[0]], 'Latitude': [gps_point[1]], 'Altitude': [gps_point[2]],
                                         'Show': [1]}))


        print(row['Time'], df.loc[index, 'X'], df.loc[index, 'Y'], df.loc[index, 'Z'])
        df_traj_gps_list.append(pd.DataFrame({'Time': [row['Time']], 'Rx(Roll)': [df.loc[index, 'Rx(Roll)']], 'Ry(Pitch)': [df.loc[index, 'Ry(Pitch)']],
                                              'Rz(Yaw)': [df.loc[index, 'Rz(Yaw)']],'X': [df.loc[index, 'X']], 'Y': [df.loc[index, 'Y']], 'Z': [df.loc[index, 'Z']],
                                              'Base Longitude': [gps_point[0]], 'Base Latitude': [gps_point[1]], 'Base Altitude': [gps_point[2]]}))


    df_gps = pd.concat(df_gps_list, ignore_index=True)
    df_traj_gps = pd.concat(df_traj_gps_list, ignore_index=True)
    df_merged = pd.concat([df_gps, df_car_base], ignore_index=True)
    return df_merged, df_traj_gps


def main():
    geo_transformer = GeoTransformer()
    df_car_base = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/car_base_merge_onto_traj_gps.csv')

    df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/v2/trajectory_v2.csv')

    df = df.drop_duplicates(subset='Time')
    df_rotated = get_slam_rotated_trajectory(df)

    GPS0 = (53.28989834, -9.07136142, 67.566)  # Origin
    GPSY = (53.28991859, -9.07135091, 67.552)  # Positive Y direction
    GPSX = (53.28991224, -9.07143945, 67.531)  # Negative X direction
    GPS_start = (53.28988778238876, -9.071366899526426, 67.57329943403602)
    df_gps, df_traj_gps = convert_slam_rotated_trajectory_to_gps(GPS0, GPSX, GPSY, GPS_start, df_rotated, df_car_base, df, geo_transformer)

    df.to_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/slam/v2/trajectory_v2.csv", index=False)
    df_rotated.to_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/slam/v2/trajectory_v2_local_cs.csv', index=False)

    df_traj_gps.to_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/slam/v2/trajectory_v2_global_cs.csv', index=False)

    df_gps.to_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/Trajectory_v2_Interpolated_GPS_Route.csv', index=False)


if __name__ == '__main__':
    main()