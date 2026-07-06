import pandas as pd
import numpy as np
from wp2.geo_utils import GeoTransformer
import nvector as nv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data():
    df_trajectory_lidar_cs = pd.read_csv(
        'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/v2/trajectory_v2.csv')
    df_car_annotations_lidar_cs = pd.read_csv(
        "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_1_annotations_car.csv")
    df_node_annotations = pd.read_csv(
        'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_1_annotations_node.csv')
    df_car_base = pd.read_csv(
        'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/car_base_merge_onto_traj_gps.csv')
    df_name_R = pd.read_csv(
        'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_with_R_for_local_to_ecef.csv')
    df_name_t_int = pd.read_csv(
        'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_for_local_to_ecef_w_SLAM_T0-Tn_Interval.csv')

    return df_trajectory_lidar_cs, df_car_annotations_lidar_cs, df_node_annotations, df_car_base, df_name_R, df_name_t_int


def preprocess_vehicle_data(geo_transformer, df_trajectory_lidar_cs, df_car_base, shift_time_index, GPS0, GPSY, GPSX,
                            GPS_start):
    df_trajectory_lidar_cs = df_trajectory_lidar_cs.drop_duplicates(subset='Time')
    df_trajectory_local_cs = geo_transformer.get_slam_rotated_trajectory(df_trajectory_lidar_cs)

    df_trajectory_gps_visualiser, df_trajectory_global_cs = geo_transformer.convert_slam_rotated_trajectory_to_gps(
        GPS0, GPSX, GPSY, GPS_start, df_trajectory_local_cs, df_car_base, df_trajectory_lidar_cs, geo_transformer
    )

    # Ok, currently, this might not be the best way to do it.
    # First, I'm getting the 'raw' df_trajectory_global_cs from above.
    # Then, I am shifting only the time column by 3 indexes (i.e., 300ms) to get spatial alignment.
    # This results in a new column beside 'Time' called 'Shifted_Time'. However, the values remain the same for everything.
    # In other words, I am only shifting the time, and not the other values like gps.
    # But this is fixed in df_car_annotations_with_vehicle_location, where I merge on shifted_time the annotations with the vehicle location.
    # See geo_localize_car_objects function.


    # Shift the time on df_trajectory_global_cs by `shift_time_index` to get spatial alignment.
    df_trajectory_global_cs['Shifted_Time'] = df_trajectory_global_cs['Time'].shift(-shift_time_index)
    df_trajectory_global_cs['Shifted_Time'] = df_trajectory_global_cs['Shifted_Time'].fillna(0)

    df_trajectory_global_cs.insert(df_trajectory_global_cs.columns.get_loc('Time') + 1, 'Shifted_Time',
                                   df_trajectory_global_cs.pop('Shifted_Time'))

    return df_trajectory_global_cs


def geo_localize_car_objects(geo_transformer, df_car_annotations_lidar_cs, df_trajectory_global_cs, df_R_t_interval):
    df_car_annotations_with_vehicle_location = pd.merge_asof(df_car_annotations_lidar_cs.sort_values('Time_Short'),
                                                             df_trajectory_global_cs.sort_values('Shifted_Time'),
                                                             left_on='Time_Short',
                                                             right_on='Shifted_Time', direction='nearest')

    df_car_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Car', 'Time_Car', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
        'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
        'Object_X_ECEF_Car', 'Object_Y_ECEF_Car', 'Object_Z_ECEF_Car'
    ])

    for index, row in df_car_annotations_with_vehicle_location.iterrows():
        row['Time'] = row['Time_Short']

        theta_lidar_to_local = np.deg2rad(-15)
        R_LidarToLocal = [[np.cos(theta_lidar_to_local), -np.sin(theta_lidar_to_local), 0],
                          [np.sin(theta_lidar_to_local), np.cos(theta_lidar_to_local), 0],
                          [0, 0, 1]]

        R_LocalToGlobal = geo_transformer.getLidarToLocalCS_Rotation(df_R_t_interval, row['Time'], geo_transformer,
                                                                     hard_values=False)

        x_off, y_off, z_off = row['X_Center'], row['Y_Center'], row['Z_Center']
        point_lidar_cs = np.array([x_off, y_off, z_off])
        point_local_cs = np.dot(R_LidarToLocal, point_lidar_cs)

        point_local_cs_left_neg = np.array([-point_local_cs[0], point_local_cs[1], point_local_cs[2]])

        GPS_base_curr = (row['Base Latitude'], row['Base Longitude'], row['Base Altitude'])
        ECEF_base_curr = geo_transformer.gps_to_ecef(*GPS_base_curr)
        point_global_cs_ecef = geo_transformer.lidar_to_ecef(point_local_cs_left_neg, ECEF_base_curr, R_LocalToGlobal)
        gps_point = geo_transformer.ecef_to_gps(*point_global_cs_ecef)

        df_car_object_geolocation = pd.concat([df_car_object_geolocation, pd.DataFrame({
            'Object_Label_Car': [row['Label']],
            'Time_Car': [row['Time_Short']],
            'Car_Latitude': [GPS_base_curr[0]],
            'Car_Longitude': [GPS_base_curr[1]],
            'Car_Altitude': [GPS_base_curr[2]],
            'Object_Latitude_Car': [gps_point[1]],
            'Object_Longitude_Car': [gps_point[0]],
            'Object_Altitude_Car': [gps_point[2]],
            'Object_X_ECEF_Car': [ECEF_base_curr[0]],
            'Object_Y_ECEF_Car': [ECEF_base_curr[1]],
            'Object_Z_ECEF_Car': [ECEF_base_curr[2]]
        })], ignore_index=True)

    return df_car_object_geolocation


def geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading):
    df_node_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Node', 'Time_Node', 'Node_Latitude', 'Node_Longitude', 'Node_Altitude',
        'Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node',
        'Object_X_ECEF_Node', 'Object_Y_ECEF_Node', 'Object_Z_ECEF_Node'
    ])

    for index, row in df_node_annotations.iterrows():
        x_off, y_off, z_off = row['X_Center'], row['Y_Center'], row['Z_Center']
        GPS_base_curr = (row['Base Latitude'], row['Base Longitude'])

        lidar_point = np.array([x_off, y_off, z_off])
        gps_point = geo_transformer.node_geolocate_object(GPS_base_curr, compass_heading, lidar_point)
        ecef_point = geo_transformer.gps_to_ecef(*gps_point)

        df_node_object_geolocation = pd.concat([df_node_object_geolocation, pd.DataFrame({
            'Object_Label_Node': [row['Label']],
            'Time_Node': [row['Time_Short']],
            'Node_Latitude': [GPS_base_curr[0]],
            'Node_Longitude': [GPS_base_curr[1]],
            'Node_Altitude': [0],
            'Object_Latitude_Node': [gps_point[1]],
            'Object_Longitude_Node': [gps_point[0]],
            'Object_Altitude_Node': [0],
            'Object_X_ECEF_Node': [ecef_point[0]],
            'Object_Y_ECEF_Node': [ecef_point[1]],
            'Object_Z_ECEF_Node': [ecef_point[2]]
        })], ignore_index=True)

    return df_node_object_geolocation


def calculate_euclidean_distances(wgs84, df_merged):
    """
    Calculate all the relevant Euclidean distances.
    """
    eucl_dist_df = pd.DataFrame(columns=[
        'Label_Car', 'Label_Node', 'Time_Car', 'Time_Node',
        'car_base_to_node_base', 'car_obj_to_node_obj'
    ])

    for _, row in df_merged.iterrows():
        car_gps = (row['Car_Latitude'], row['Car_Longitude'], row['Car_Altitude'])
        node_gps = (row['Node_Latitude'], row['Node_Longitude'], row['Node_Altitude'])
        obj_car_gps = (row['Object_Latitude_Car'], row['Object_Longitude_Car'], row['Object_Altitude_Car'])
        obj_node_gps = (row['Object_Latitude_Node'], row['Object_Longitude_Node'], row['Object_Altitude_Node'])

        point_car_gps = wgs84.GeoPoint(latitude=car_gps[0], longitude=car_gps[1], degrees=True)
        point_node_gps = wgs84.GeoPoint(latitude=node_gps[0], longitude=node_gps[1], degrees=True)
        point_obj_car_gps = wgs84.GeoPoint(latitude=obj_car_gps[0], longitude=obj_car_gps[1], degrees=True)
        point_obj_node_gps = wgs84.GeoPoint(latitude=obj_node_gps[0], longitude=obj_node_gps[1], degrees=True)

        car_base_to_node_base = point_car_gps.to_ecef_vector() - point_node_gps.to_ecef_vector()
        car_obj_to_node_obj = point_obj_car_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()
        car_base_to_car_obj = point_car_gps.to_ecef_vector() - point_obj_car_gps.to_ecef_vector()
        node_base_to_node_obj = point_node_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()
        car_base_to_node_obj = point_car_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()
        node_base_to_car_obj = point_node_gps.to_ecef_vector() - point_obj_car_gps.to_ecef_vector()

        dist_eucl_car_base_to_node_base = car_base_to_node_base.length
        dist_eucl_car_obj_to_node_obj = car_obj_to_node_obj.length
        dist_eucl_car_base_to_car_obj = car_base_to_car_obj.length
        dist_eucl_node_base_to_node_obj = node_base_to_node_obj.length
        dist_eucl_car_base_to_node_obj = car_base_to_node_obj.length
        dist_eucl_node_base_to_car_obj = node_base_to_car_obj.length

        eucl_dist_row = pd.DataFrame({
            'Label_Car': [row['Object_Label_Car']],
            'Label_Node': [row['Object_Label_Node']],
            'Time_Car': [row['Time_Car']],
            'Time_Node': [row['Time_Node']],
            'car_base_to_node_base': [dist_eucl_car_base_to_node_base],
            'car_obj_to_node_obj': [dist_eucl_car_obj_to_node_obj],
            'car_base_to_car_obj': [dist_eucl_car_base_to_car_obj],
            'node_base_to_node_obj': [dist_eucl_node_base_to_node_obj],
            'car_base_to_node_obj': [dist_eucl_car_base_to_node_obj],
            'node_base_to_car_obj': [dist_eucl_node_base_to_car_obj]
        })

        eucl_dist_df = pd.concat([eucl_dist_df, eucl_dist_row], ignore_index=True)

    return eucl_dist_df


def main():
    geo_transformer = GeoTransformer()
    wgs84 = nv.FrameE(name='WGS84')

    # Constants
    GPS0 = (53.28989834, -9.07136142, 67.566)
    GPSY = (53.28991859, -9.07135091, 67.552)
    GPSX = (53.28991224, -9.07143945, 67.531)
    GPS_start = (53.28988778238876, -9.071366899526426, 67.57329943403602)
    compass_heading = 245.1

    # Load data
    df_trajectory_lidar_cs, df_car_annotations_lidar_cs, df_node_annotations, df_car_base, df_name_R, df_name_t_int = load_data()

    # Combine rotation matrix and time interval info
    df_R_t_interval = pd.merge(df_name_R, df_name_t_int, on='Base_Point')

    # Specify shift_time_index values
    shift_time_indices = [3, 13]  # Custom indices to represent the shift

    # Keep track of distances at index 0 for second plot
    distance_at_index_0 = []

    ### FIRST PLOT: Euclidean Distance Between Car Objects and Node Objects Over Time ###
    plt.figure(figsize=(10, 6))

    for shift_time_index in shift_time_indices:
        # Preprocess vehicle data
        df_trajectory_global_cs = preprocess_vehicle_data(geo_transformer, df_trajectory_lidar_cs, df_car_base,
                                                          shift_time_index, GPS0, GPSY, GPSX, GPS_start)

        # Geolocate car and node objects
        df_car_object_geolocation = geo_localize_car_objects(geo_transformer, df_car_annotations_lidar_cs,
                                                             df_trajectory_global_cs, df_R_t_interval)
        df_node_object_geolocation = geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading)

        # Merge geolocated car and node data
        df_merged_node_car_object = pd.concat([df_car_object_geolocation.reset_index(drop=True),
                                               df_node_object_geolocation.reset_index(drop=True)], axis=1)

        # Calculate distances
        eucl_dist_df = calculate_euclidean_distances(wgs84, df_merged_node_car_object)

        # Calculate the timestamp jitter (in ms), mapped from (shift_time_index - 3)
        jitter = (shift_time_index - 3) * 1  # 1ms per index after 3

        # Plot Euclidean distance between car objects and node objects over time
        if not eucl_dist_df.empty:
            plt.plot(eucl_dist_df['Time_Car'], eucl_dist_df['car_obj_to_node_obj'],
                     marker='o', linestyle='-',
                     label=f'timestamp_jitter={jitter}s')

    plt.xlabel('Sensor Timestamp')
    plt.ylabel('Distance (m)')
    plt.title('Euclidean Distance Between Car Detected Objects and Node Detected Objects Over Time\n Object = Person 1')
    plt.legend()
    plt.grid(True)
    # Set major ticks
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Enable minor ticks
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    # Add grid for minor ticks
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    # Set formatter for x and y-axis
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    plt.show()

    ### SECOND PLOT: Distance at Index 0 vs shift_time_index ###
    plt.figure(figsize=(10, 6))

    for shift_time_index in shift_time_indices:
        # In case we want to aggregate this for the second plot
        df_trajectory_global_cs = preprocess_vehicle_data(geo_transformer, df_trajectory_lidar_cs, df_car_base,
                                                          shift_time_index, GPS0, GPSY, GPSX, GPS_start)

        # Geolocate objects
        df_car_object_geolocation = geo_localize_car_objects(geo_transformer, df_car_annotations_lidar_cs,
                                                             df_trajectory_global_cs, df_R_t_interval)
        df_node_object_geolocation = geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading)

        # Merge DataFrames
        df_merged_node_car_object = pd.concat([df_car_object_geolocation.reset_index(drop=True),
                                               df_node_object_geolocation.reset_index(drop=True)], axis=1)

        # Calculate distances and extract index 0
        eucl_dist_df = calculate_euclidean_distances(wgs84, df_merged_node_car_object)

        # Append the distance at index 0 of the dataframe (`car_obj_to_node_obj`)
        if not eucl_dist_df.empty:
            distance_at_index_0.append(eucl_dist_df['car_obj_to_node_obj'].iloc[0])

    # Plot shift_time_index against the distance at index 0
    plt.plot(shift_time_indices, distance_at_index_0, marker='o', linestyle='-', color='black', label='Distance at Index 0')
    plt.xlabel('Timestamp Jitter (s)')
    plt.ylabel('Car_Obj_to_Node_Obj Distance (m)')
    plt.title('Distance at Index 0 vs Timestamp Jitter (s)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()