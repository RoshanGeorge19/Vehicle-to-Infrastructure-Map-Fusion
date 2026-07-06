import pandas as pd
import numpy as np
from wp2.geo_utils import GeoTransformer, Plotter
import nvector as nv
import pickle


def load_data():
    # Important Data.
    df_trajectory_lidar_cs = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/v2/trajectory_v2.csv')
    df_car_annotations_lidar_cs = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_1_annotations_car.csv")
    df_node_annotations = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_1_annotations_node.csv')
    # Supplementary Data.
    df_car_base = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/car_base_merge_onto_traj_gps.csv')
    df_name_R = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_with_R_for_local_to_ecef.csv')
    df_name_t_int = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_for_local_to_ecef_w_SLAM_T0-Tn_Interval.csv')
    # Merged R matrix, and time interval from df_name_R and df_name_t_int.
    df_R_t_interval = pd.merge(df_name_R, df_name_t_int, on='Base_Point')
    # Constants
    GPS0 = (53.28989834, -9.07136142, 67.566)
    GPSY = (53.28991859, -9.07135091, 67.552)
    GPSX = (53.28991224, -9.07143945, 67.531)
    GPS_start = (53.28988778238876, -9.071366899526426, 67.57329943403602)
    compass_heading = 245.1

    return df_trajectory_lidar_cs, df_car_annotations_lidar_cs, df_node_annotations, df_car_base, df_R_t_interval, GPS0, GPSY, GPSX, GPS_start, compass_heading

def preprocess_vehicle_data(geo_transformer, df_trajectory_lidar_cs, df_car_base, delay, GPS0, GPSY, GPSX, GPS_start):

    df_trajectory_lidar_cs = df_trajectory_lidar_cs.drop_duplicates(subset='Time')
    df_trajectory_local_cs = geo_transformer.get_slam_rotated_trajectory(df_trajectory_lidar_cs)

    df_trajectory_gps_visualiser, df_trajectory_global_cs = geo_transformer.convert_slam_rotated_trajectory_to_gps(
        GPS0, GPSX, GPSY, GPS_start, df_trajectory_local_cs, df_car_base, df_trajectory_lidar_cs, geo_transformer)

    time_constant_shift = 300 # 300 ms shift to ensure spatial alignment.
    # Should you minus or add the delay?
    shift_time_index = int((time_constant_shift / 100) - (delay / 100))

    # Shift the time on df_trajectory_global_cs by `shift_time_index` to get spatial alignment.
    df_trajectory_global_cs['Shifted_Time'] = df_trajectory_global_cs['Time'].shift(-shift_time_index)
    df_trajectory_global_cs['Shifted_Time'] = df_trajectory_global_cs['Shifted_Time'].fillna(0)

    df_trajectory_global_cs.insert(df_trajectory_global_cs.columns.get_loc('Time') + 1, 'Shifted_Time',
                                   df_trajectory_global_cs.pop('Shifted_Time'))

    return df_trajectory_global_cs

def preprocess_node_data(geo_transformer, df_node_annotations, delay, compass_heading):
    df_node_annotations_old = df_node_annotations.copy() # For debugging purposes. This is without the shift.

    df_node_annotations = df_node_annotations.drop(columns=['File_Name', 'Time'])
    df_node_annotations = df_node_annotations.rename(columns={'Time_Short': 'Time'})

    time_constant_shift = 0  # In ms. Already aligned, so no constant time shift needed.
    shift_time_index = int((time_constant_shift / 100) + (delay / 100))

    # Shift the time on df_trajectory_global_cs by `shift_time_index` to get spatial alignment.
    df_node_annotations['Shifted_Time'] = df_node_annotations['Time'].shift(shift_time_index)
    df_node_annotations['Shifted_Time'] = df_node_annotations['Shifted_Time'].fillna(0)

    df_node_annotations.insert(df_node_annotations.columns.get_loc('Time') + 1, 'Shifted_Time', df_node_annotations.pop('Shifted_Time'))

    # df_node_annotations = df_node_annotations.shift(shift_time_index).reset_index(drop=True)
    # df_node_annotations = df_node_annotations.fillna(0)
    # df_node_annotations['Shifted_Time'] = df_node_annotations['Time'].shift(shift_time_index).reset_index(drop=True).fillna(0)
    # df_node_annotations['Time'] = df_node_annotations['Time']
    # df_node_annotations.insert(df_node_annotations.columns.get_loc('Time') + 1, 'Shifted_Time', df_node_annotations.pop('Shifted_Time'))

    return df_node_annotations

def geo_localize_car_objects_old(geo_transformer, df_car_annotations_lidar_cs, df_trajectory_global_cs, df_R_t_interval):
    df_car_annotations_with_vehicle_location = pd.merge_asof(df_car_annotations_lidar_cs.sort_values('Time_Short'),
                                                             df_trajectory_global_cs.sort_values('Shifted_Time'),
                                                             left_on='Time_Short',
                                                             right_on='Shifted_Time', direction='nearest')

    df_car_annotations_with_vehicle_location['Diff'] = df_car_annotations_with_vehicle_location['Time_Short'] - df_car_annotations_with_vehicle_location['Shifted_Time']

    # Manually set the columns
    df_car_annotations_with_vehicle_location = pd.DataFrame(
        df_car_annotations_with_vehicle_location,
        columns=['File_Name', 'Label', 'Shifted_Time', 'Time_Short', 'Diff', 'X_Center', 'Y_Center', 'Z_Center',
                 'Length', 'Width', 'Height', 'Rx', 'Ry', 'Rz', 'Base Latitude', 'Base Longitude', 'Base Altitude',
                 'X', 'Y', 'Z', 'Rx(Roll)', 'Ry(Pitch)', 'Rz(Yaw)'])

    df_car_annotations_with_vehicle_location_old = df_car_annotations_with_vehicle_location.copy()

    duplicate_value_in_shifted_time = df_car_annotations_with_vehicle_location[df_car_annotations_with_vehicle_location.duplicated(subset='Shifted_Time', keep=False)]
    positive_diff_rows = df_car_annotations_with_vehicle_location[(df_car_annotations_with_vehicle_location.duplicated(subset='Shifted_Time', keep=False))
                                                                  & (df_car_annotations_with_vehicle_location['Diff'] > 0)]

    df_trajectory_global_cs_sorted = df_trajectory_global_cs.sort_values('Shifted_Time').reset_index(drop=True)
    updated_shifted_times = []

    for index, row in positive_diff_rows.iterrows():
        current_shifted_time = row['Shifted_Time']
        next_times = df_trajectory_global_cs_sorted[df_trajectory_global_cs_sorted['Shifted_Time'] > current_shifted_time]
        if not next_times.empty:
            next_time = next_times.iloc[0]['Shifted_Time']
        else:
            next_time = float('nan')
        updated_shifted_times.append(next_time)

    updated_rows = []
    # Loop through the updated_shifted_times (for the positive_diff_rows)
    for next_shifted_time in updated_shifted_times:
        # Find the row in 'df_trajectory_global_cs' where 'Shifted_Time' == next_shifted_time
        new_row = df_trajectory_global_cs_sorted[df_trajectory_global_cs_sorted['Shifted_Time'] == next_shifted_time]

        if not new_row.empty:
            # Append this new row from 'df_trajectory_global_cs' to our list of updated rows
            updated_rows.append(new_row)
        else:
            # If no matching row found (for NaN case), skip or handle accordingly
            updated_rows.append(
                pd.DataFrame())  # Add an empty DataFrame as a placeholder (or handle this case separately)

    updated_rows_df = pd.concat(updated_rows).reset_index(drop=True)

    # Find the index of the rows you want to update
    rows_to_update_idx = positive_diff_rows.index

    # Ensure 'updated_rows_df' holds the correct updated values
    # Now, update only the rows with the intended index from 'positive_diff_rows'
    df_car_annotations_with_vehicle_location.loc[rows_to_update_idx, updated_rows_df.columns] = updated_rows_df.values

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
            'Time_Car': [row['Shifted_Time']], # Note this is the shifted time now. Not the original time on the file_name. i.e., 1099.700.
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
        print(f"1 CAR_0ms_Off_T{index} {gps_point[0]} {gps_point[1]} {gps_point[2]} 1")

    return df_car_object_geolocation

def geo_localize_car_objects(geo_transformer, df_car_annotations_lidar_cs, df_trajectory_global_cs, df_R_t_interval):
    # Initial nearest match merging based on Time_Short and Shifted_Time
    df_car_annotations_with_vehicle_location = pd.merge_asof(
        df_car_annotations_lidar_cs.sort_values('Time_Short'),
        df_trajectory_global_cs.sort_values('Shifted_Time'),
        left_on='Time_Short',
        right_on='Shifted_Time',
        direction='nearest'
    )

    # Calculate the time difference (Diff)
    df_car_annotations_with_vehicle_location['Diff'] = (df_car_annotations_with_vehicle_location['Time_Short'] - df_car_annotations_with_vehicle_location['Shifted_Time'])

    # Set the columns
    df_car_annotations_with_vehicle_location = pd.DataFrame(
        df_car_annotations_with_vehicle_location,
        columns=['File_Name', 'Label', 'Shifted_Time', 'Time_Short', 'Diff', 'X_Center', 'Y_Center', 'Z_Center',
                 'Length', 'Width', 'Height', 'Rx', 'Ry', 'Rz', 'Base Latitude', 'Base Longitude', 'Base Altitude',
                 'X', 'Y', 'Z', 'Rx(Roll)', 'Ry(Pitch)', 'Rz(Yaw)']
    )

    df_car_annotations_with_vehicle_location_old = df_car_annotations_with_vehicle_location.copy()

    #### Recursive Duplicate Handling ####
    # Keep running until no duplicates exist in the 'Shifted_Time' column.
    while not df_car_annotations_with_vehicle_location['Shifted_Time'].duplicated().sum() == 0:
        # Identify rows with duplicated 'Shifted_Time' and positive 'Diff'
        positive_diff_rows = df_car_annotations_with_vehicle_location[
            (df_car_annotations_with_vehicle_location.duplicated(subset='Shifted_Time', keep=False)) &
            (df_car_annotations_with_vehicle_location['Diff'] > 0)]

        if positive_diff_rows.empty:
            print("No more duplicates with positive Diff exist.")
            break  # If there are no rows left to update, exit the loop

        # Sort the trajectory data for lookup
        df_trajectory_global_cs_sorted = df_trajectory_global_cs.sort_values('Shifted_Time').reset_index(drop=True)

        # Collect updated Shifted_Time values
        updated_shifted_times = []
        for index, row in positive_diff_rows.iterrows():
            current_shifted_time = row['Shifted_Time']

            # Find the next closest Shifted_Time greater than the current value
            next_times = df_trajectory_global_cs_sorted[
                df_trajectory_global_cs_sorted['Shifted_Time'] > current_shifted_time]
            if not next_times.empty:
                next_time = next_times.iloc[0]['Shifted_Time']  # Take the first next time
            else:
                next_time = float('nan')  # Handle with NaN, if no future Shifted_Time found
            updated_shifted_times.append(next_time)

        # Fetch full rows from df_trajectory_global_cs corresponding to next Shifted_Time
        updated_rows = []
        for next_shifted_time in updated_shifted_times:
            new_row = df_trajectory_global_cs_sorted[
                df_trajectory_global_cs_sorted['Shifted_Time'] == next_shifted_time]
            if not new_row.empty:
                updated_rows.append(new_row)
            else:
                updated_rows.append(pd.DataFrame())  # Handle NaN (empty) case appropriately

        # Concatenate to create the updated DataFrame
        updated_rows_df = pd.concat(updated_rows).reset_index(drop=True)

        # Update the original DataFrame based on the index in positive_diff_rows
        rows_to_update_idx = positive_diff_rows.index
        df_car_annotations_with_vehicle_location.loc[
            rows_to_update_idx, updated_rows_df.columns] = updated_rows_df.values

        # Recalculate 'Diff' after updates
        df_car_annotations_with_vehicle_location['Diff'] = (df_car_annotations_with_vehicle_location['Time_Short'] - df_car_annotations_with_vehicle_location['Shifted_Time'])
        df_car_annotations_with_vehicle_location = df_car_annotations_with_vehicle_location.drop(columns=['Time'])
    #### End of Recursive Duplicate Handling ####



    # Create output DataFrame for car object geolocation
    df_car_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Car', 'Time_Car', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
        'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
        'Object_X_ECEF_Car', 'Object_Y_ECEF_Car', 'Object_Z_ECEF_Car'
    ])

    # Iterate through rows in the updated DataFrame for geolocation
    for index, row in df_car_annotations_with_vehicle_location.iterrows():
        row['Time'] = row['Time_Short']

        # Rotation matrix to convert from Lidar to local coordinate system
        theta_lidar_to_local = np.deg2rad(-15)
        R_LidarToLocal = [[np.cos(theta_lidar_to_local), -np.sin(theta_lidar_to_local), 0],
                          [np.sin(theta_lidar_to_local), np.cos(theta_lidar_to_local), 0],
                          [0, 0, 1]]

        # Get the rotation matrix from local to global
        R_LocalToGlobal = geo_transformer.getLidarToLocalCS_Rotation(
            df_R_t_interval, row['Time'], geo_transformer, hard_values=False
        )

        # Process X, Y, Z values for geolocation
        x_off, y_off, z_off = row['X_Center'], row['Y_Center'], row['Z_Center']
        point_lidar_cs = np.array([x_off, y_off, z_off])
        point_local_cs = np.dot(R_LidarToLocal, point_lidar_cs)

        point_local_cs_left_neg = np.array([-point_local_cs[0], point_local_cs[1], point_local_cs[2]])

        # Convert GPS Coordinates
        GPS_base_curr = (row['Base Latitude'], row['Base Longitude'], row['Base Altitude'])
        ECEF_base_curr = geo_transformer.gps_to_ecef(*GPS_base_curr)
        point_global_cs_ecef = geo_transformer.lidar_to_ecef(point_local_cs_left_neg, ECEF_base_curr, R_LocalToGlobal)
        gps_point = geo_transformer.ecef_to_gps(*point_global_cs_ecef)

        # Append the geolocation details to the resulting dataframe
        df_car_object_geolocation = pd.concat([df_car_object_geolocation, pd.DataFrame({
            'Object_Label_Car': [row['Label']],
            'Time_Car': [row['Shifted_Time']],  # Using Shifted_Time after updates
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
        print(f"1 CAR_0ms_Off_T{index} {gps_point[0]} {gps_point[1]} {gps_point[2]} 1")

    return df_car_object_geolocation

def geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading, delay):
    df_node_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Node', 'Time_Node', 'Time_Node_Shifted', 'Node_Latitude', 'Node_Longitude', 'Node_Altitude',
        'Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node',
        'Object_X_ECEF_Node', 'Object_Y_ECEF_Node', 'Object_Z_ECEF_Node',
        'X', 'Y', 'Z'
    ])

    for index, row in df_node_annotations.iterrows():
        shifted_time = row['Shifted_Time']

        if shifted_time == 0:
            # continue
            df_node_object_geolocation = pd.concat([df_node_object_geolocation, pd.DataFrame({
                'Object_Label_Node': [row['Label']],
                'Time_Node': [row['Time']],
                'Time_Node_Shifted': [row['Shifted_Time']],
                # This is the shifted time, i.e, the time with the delay of processing.
                'Node_Latitude': [row['Base Latitude']],
                'Node_Longitude': [row['Base Longitude']],
                'Node_Altitude': [0],
                'Object_Latitude_Node': [row['Base Latitude']],
                'Object_Longitude_Node': [row['Base Longitude']],
                'Object_Altitude_Node': [0],
                'Object_X_ECEF_Node': [0],
                'Object_Y_ECEF_Node': [0],
                'Object_Z_ECEF_Node': [0],
                'X': [0],
                'Y': [0],
                'Z': [0]
            })], ignore_index=True)
            print(f"200 Node_{delay}ms_Off_T{index} {row['Base Longitude']} {row['Base Latitude']} 0 1")

        else:
            matching_row = df_node_annotations[df_node_annotations['Time'] == shifted_time].iloc[0]
            x_off, y_off, z_off = matching_row['X_Center'], matching_row['Y_Center'], matching_row['Z_Center']
            GPS_base_curr = (matching_row['Base Latitude'], matching_row['Base Longitude'])

            lidar_point = np.array([x_off, y_off, z_off])
            gps_point = geo_transformer.node_geolocate_object(GPS_base_curr, compass_heading, lidar_point)
            ecef_point = geo_transformer.gps_to_ecef(*gps_point)

            df_node_object_geolocation = pd.concat([df_node_object_geolocation, pd.DataFrame({
                'Object_Label_Node': [row['Label']],
                'Time_Node': [row['Time']],
                'Time_Node_Shifted': [row['Shifted_Time']], # This is the shifted time, i.e, the time with the delay of processing.
                'Node_Latitude': [GPS_base_curr[0]],
                'Node_Longitude': [GPS_base_curr[1]],
                'Node_Altitude': [0],
                'Object_Latitude_Node': [gps_point[1]],
                'Object_Longitude_Node': [gps_point[0]],
                'Object_Altitude_Node': [0],
                'Object_X_ECEF_Node': [ecef_point[0]],
                'Object_Y_ECEF_Node': [ecef_point[1]],
                'Object_Z_ECEF_Node': [ecef_point[2]],
                'X': [matching_row['X_Center']],
                'Y': [matching_row['Y_Center']],
                'Z': [matching_row['Z_Center']]
            })], ignore_index=True)
            print(f"2 Node_{delay}ms_Off_T{index} {gps_point[0]} {gps_point[1]} 0 1")


        # df_node_object_geolocation = pd.concat([df_node_object_geolocation, pd.DataFrame({
        #     'Object_Label_Node': [row['Label']],
        #     'Time_Node': [row['Time']],
        #     'Time_Node_Shifted': [row['Shifted_Time']],
        #     'X': [matching_row['X_Center']],
        #     'Y': [matching_row['Y_Center']],
        #     'Z': [matching_row['Z_Center']]
        # })], ignore_index=True)

    return df_node_object_geolocation

def calculate_euclidean_distances(wgs84, df_merged):

    eucl_dist_df = pd.DataFrame(columns=[
        'Label_Car', 'Label_Node', 'Time_Car', 'Time_Node', 'Time_Node_Shifted',
        'car_base_to_node_base', 'car_obj_to_node_obj', 'car_base_to_car_obj', 'node_base_to_node_obj',
        'car_base_to_node_obj', 'node_base_to_car_obj'
    ])

    for column, row in df_merged.iterrows():
        small_threshold = 1e-6
        if abs(row['Time_Node_Shifted']) > small_threshold:
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

        else:
            dist_eucl_car_base_to_node_base = np.nan
            dist_eucl_car_obj_to_node_obj = np.nan
            dist_eucl_car_base_to_car_obj = np.nan
            dist_eucl_node_base_to_node_obj = np.nan
            dist_eucl_car_base_to_node_obj = np.nan
            dist_eucl_node_base_to_car_obj = np.nan

        eucl_dist_row = pd.DataFrame({
            'Label_Car': [row['Object_Label_Car']],
            'Label_Node': [row['Object_Label_Node']],
            'Time_Car': [row['Time_Car']],
            'Time_Node': [row['Time_Node']],
            'Time_Node_Shifted': [row['Time_Node_Shifted']],
            'car_base_to_node_base': [dist_eucl_car_base_to_node_base],
            'car_obj_to_node_obj': [dist_eucl_car_obj_to_node_obj],
            'car_base_to_car_obj': [dist_eucl_car_base_to_car_obj],
            'node_base_to_node_obj': [dist_eucl_node_base_to_node_obj],
            'car_base_to_node_obj': [dist_eucl_car_base_to_node_obj],
            'node_base_to_car_obj': [dist_eucl_node_base_to_car_obj]
        })
        eucl_dist_df = pd.concat([eucl_dist_df, eucl_dist_row], ignore_index=True)

    return eucl_dist_df

def getRelativeError(eucl_dist_dict):
    relative_error_dict = {}  # Dictionary to store the relative error DataFrames

    sorted_delays = sorted(eucl_dist_dict.keys())  # Make sure delays are sorted in ascending order

    # Iterate over the sorted delays, comparing consecutive delay DataFrames
    for i in range(len(sorted_delays) - 1):
        delay = sorted_delays[i]
        next_delay = sorted_delays[i + 1]

        # Get the DataFrames for the current and the next delay
        df_current = eucl_dist_dict[delay]
        df_next = eucl_dist_dict[next_delay]

        # Prepare a DataFrame to hold relative errors for the current and next delay
        relative_error_df = pd.DataFrame(columns=['Label_Car', 'Label_Node', 'Time_Car', 'Time_Node',
                                                  'Time_Node_Shifted', 'car_obj_to_node_obj', 'rel_error'])

        # Ensure both DataFrames are sorted by 'Time_Car' or 'Time_Node' to align rows
        df_current = df_current.sort_values(by='Time_Car').reset_index(drop=True)
        df_next = df_next.sort_values(by='Time_Car').reset_index(drop=True)

        # Take the minimum number of rows to handle different DataFrame lengths
        min_length = min(len(df_current), len(df_next))

        for j in range(min_length):
            # Extract the 'car_obj_to_node_obj' values from both consecutive delays
            car_obj_to_node_obj_current = df_current.loc[j, 'car_obj_to_node_obj']
            car_obj_to_node_obj_next = df_next.loc[j, 'car_obj_to_node_obj']

            # Compute the relative error (difference between the consecutive delay values)
            # relative_error = car_obj_to_node_obj_next - car_obj_to_node_obj_current
            relative_error = abs(car_obj_to_node_obj_current - car_obj_to_node_obj_next)

            # Append to the relative_error_df DataFrame
            relative_error_df = pd.concat([relative_error_df, pd.DataFrame({
                'Label_Car': [df_current.loc[j, 'Label_Car']],
                'Label_Node': [df_current.loc[j, 'Label_Node']],
                'Time_Car': [df_current.loc[j, 'Time_Car']],
                'Time_Node': [df_current.loc[j, 'Time_Node']],
                'Time_Node_Shifted': [df_current.loc[j, 'Time_Node_Shifted']],
                'car_obj_to_node_obj': [car_obj_to_node_obj_current],
                'rel_error': [relative_error]
            })], ignore_index=True)

        # Store the relative error DataFrame keyed by (delay, next_delay)
        relative_error_dict[delay] = relative_error_df

    return relative_error_dict

# def calculate_difference_on_relative_error(eucl_dist_dict):
#     diff_df = pd.DataFrame(columns=['Time_Car', 'Label_Car', 'Label_Node'] +
#                                    ['Diff_' + str(sorted_delay) + '-' + str(sorted_delay + 100) for sorted_delay in
#                                     sorted(eucl_dist_dict.keys())[:-1]])
#
#     # Get complete sorted delays
#     sorted_delays = sorted(eucl_dist_dict.keys())
#
#     # Iterate over all Time_Car keys since they're common in both dataframes across delays
#     for label_node in eucl_dist_dict[0]['Label_Node'].unique():
#         for time_car in eucl_dist_dict[0]['Time_Car'].unique():
#
#             # Rows for each delay at specific Time_Car and Label_Node
#             rows_at_time_car = {
#                 delay: eucl_dist_dict[delay][(eucl_dist_dict[delay]['Time_Car'] == time_car) &
#                                              (eucl_dist_dict[delay]['Label_Node'] == label_node)]
#                 for delay in sorted_delays
#             }
#
#             # Ensure there are valid entries across all delays
#             if all(not rows_at_time_car[delay].empty for delay in sorted_delays):
#
#                 # Initialize list to store differences for this time_car over sorted_delays
#                 diff_list = []
#
#                 # Iterate over consecutive delays and compute A-B, B-C, C-D
#                 for i in range(len(sorted_delays) - 1):
#                     current_delay = sorted_delays[i]
#                     next_delay = sorted_delays[i + 1]
#
#                     # Compute 'car_obj_to_node_obj' for consecutive delays and take the difference
#                     current_error = rows_at_time_car[current_delay]['car_obj_to_node_obj'].values[0]
#                     next_error = rows_at_time_car[next_delay]['car_obj_to_node_obj'].values[0]
#
#                     # Add the difference (absolute or relative difference can be used)
#                     diff_list.append(next_error - current_error)
#
#                 # Create a row for this Time_Car to be added into the diff_df
#                 diff_df = pd.concat([diff_df, pd.DataFrame({
#                     'Time_Car': [time_car],
#                     'Label_Car': [rows_at_time_car[sorted_delays[0]]['Label_Car'].values[0]],
#                     'Label_Node': [label_node],
#                     **{'Diff_' + str(sorted_delays[i]) + '-' + str(sorted_delays[i + 1]): [diff_list[i]]
#                        for i in range(len(sorted_delays) - 1)}
#                 })], ignore_index=True)
#
#     return diff_df

def calculate_difference_on_relative_error(eucl_dist_dict):
    # Example.
    # Delay 0, 100, 200, 300, 400, 500.
    # Rel. Error = 100-0, 200-0, 300-0, 400-0, 500-0.
    # Initialize the output DataFrame with appropriate columns.
    diff_df = pd.DataFrame(columns=['Time_Car', 'Label_Car', 'Label_Node'] +
                                   ['Diff_' + str(0) + '-' + str(delay) for delay in sorted(eucl_dist_dict.keys()) if delay != 0])

    # Get a sorted list of delays
    sorted_delays = sorted(eucl_dist_dict.keys())

    # Ensure the baseline (0ms delay) exists in the dictionary
    if 0 not in sorted_delays:
        raise ValueError("Key 0 (baseline) must be present in `eucl_dist_dict`.")

    # Iterate over all unique Label_Node and Time_Car combinations in the baseline data (key 0)
    for label_node in eucl_dist_dict[0]['Label_Node'].unique():
        for time_car in eucl_dist_dict[0]['Time_Car'].unique():

            # Row for the given Label_Node and Time_Car in the baseline (0ms delay)
            baseline_row = eucl_dist_dict[0][(eucl_dist_dict[0]['Time_Car'] == time_car) &
                                             (eucl_dist_dict[0]['Label_Node'] == label_node)]

            # Check if the baseline data exists for this combination
            if not baseline_row.empty:
                # Extract the baseline value for 'car_obj_to_node_obj'
                baseline_error = baseline_row['car_obj_to_node_obj'].values[0]

                # Initialize the difference dictionary for this combination
                diff_data = {
                    'Time_Car': time_car,
                    'Label_Car': baseline_row['Label_Car'].values[0],
                    'Label_Node': label_node,
                }

                # Iterate over all delays except the baseline (0ms delay)
                for delay in sorted_delays:
                    if delay != 0:
                        # Row for the given Label_Node and Time_Car at the current delay
                        current_row = eucl_dist_dict[delay][(eucl_dist_dict[delay]['Time_Car'] == time_car) &
                                                            (eucl_dist_dict[delay]['Label_Node'] == label_node)]

                        # Check if data exists for this delay
                        if not current_row.empty:
                            # Extract the value for 'car_obj_to_node_obj' at the current delay
                            current_error = current_row['car_obj_to_node_obj'].values[0]

                            # Compute the absolute difference between the baseline and the current delay
                            diff_data['Diff_' + str(0) + '-' + str(delay)] = abs(current_error - baseline_error)
                        else:
                            # If data is missing for this delay, set the difference to NaN
                            diff_data['Diff_' + str(0) + '-' + str(delay)] = None

                # Append the computed difference data to the output DataFrame
                diff_df = pd.concat([diff_df, pd.DataFrame([diff_data])], ignore_index=True)

    return diff_df

def get_error_metrics(eucl_dist_df):
    k = 1 # error ranges from 0 to 1
    d_max = 5 # max distance, where u(d) reaches 1.
    a = -np.log(0.01)/d_max
    small_threshold = 1e-6


    error_metrics_df = pd.DataFrame(columns=[
        'Label_Car', 'Label_Node', 'Time_Car', 'Time_Node',
        'car_base_to_car_obj', 'node_base_to_node_obj', 'car_obj_to_node_obj', 'metric_1', 'metric_2'
    ])

    for index, row in eucl_dist_df.iterrows():
        d_car_obj_to_node_obj = row['car_obj_to_node_obj']
        metric_1 = (k * (1 - np.exp(-a * d_car_obj_to_node_obj)))

        d_car_base_to_car_obj = row['car_base_to_car_obj']
        d_node_base_to_node_obj = row['node_base_to_node_obj']
        weight_car = 1 / d_car_base_to_car_obj if d_car_base_to_car_obj > small_threshold else 1e-6
        weight_node = 1 / d_node_base_to_node_obj if d_node_base_to_node_obj > small_threshold else 1e-6
        weight_total = weight_car + weight_node
        metric_2 = ((weight_car)*(k * (1 - np.exp(-a * d_car_obj_to_node_obj)))) + ((weight_node)*(k * (1 - np.exp(-a * d_car_obj_to_node_obj))))


        error_metric_row = pd.DataFrame({
            'Label_Car': [row['Label_Car']],
            'Label_Node': [row['Label_Node']],
            'Time_Car': [row['Time_Car']],
            'Time_Node': [row['Time_Node']],
            'car_base_to_car_obj': row['car_base_to_car_obj'],
            'node_base_to_node_obj': row['node_base_to_node_obj'],
            'car_obj_to_node_obj': row['car_obj_to_node_obj'],
            'metric_1': [metric_1],
            'metric_2': [metric_2]})
        error_metrics_df = pd.concat([error_metrics_df, error_metric_row], ignore_index=True)
    return error_metrics_df

def main():
    geo_transformer = GeoTransformer()
    plotter = Plotter()
    wgs84 = nv.FrameE(name='WGS84')
    df_trajectory_lidar_cs, df_car_annotations_lidar_cs, df_node_annotations_orig, df_car_base, df_R_t_interval, GPS0, GPSY, GPSX, GPS_start, compass_heading = load_data()

    delay_car_list = [0]
    # delay_node_list = np.arange(0, 1000, 500)
    delay_node_list = [0]
    eucl_dist_dict = {}
    error_metrics_dict = {}
    df_merged_node_car_object_dict = {}
    df_node_object_geolocation_dict = {}
    df_car_object_geolocation_dict = {}

    for delay_car in delay_car_list:
        for delay_node in delay_node_list:
            df_trajectory_global_cs = preprocess_vehicle_data(geo_transformer, df_trajectory_lidar_cs, df_car_base, delay_car, GPS0, GPSY, GPSX, GPS_start)
            df_node_annotations = preprocess_node_data(geo_transformer, df_node_annotations_orig, delay_node, compass_heading)

            df_car_object_geolocation = geo_localize_car_objects(geo_transformer, df_car_annotations_lidar_cs, df_trajectory_global_cs, df_R_t_interval)
            df_car_object_geolocation_dict[delay_node] = df_car_object_geolocation_dict  # Store the result in the dictionary, keyed by delay_node
            pd.options.display.float_format = '{:.10f}'.format

            print(df_car_object_geolocation.to_string())
            break
            print('\n')
            df_node_object_geolocation = geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading, delay_node)
            df_node_object_geolocation_dict[delay_node] = df_node_object_geolocation  # Store the result in the dictionary, keyed by delay_node

            df_merged_node_car_object = pd.merge_asof(
                df_car_object_geolocation.sort_values('Time_Car'),  # Sort by Time_Car
                df_node_object_geolocation.sort_values('Time_Node'),  # Sort by Time_Node
                left_on='Time_Car',  # Merge on Time_Car from df_car_object_geolocation
                right_on='Time_Node',  # Merge on Time_Node_Shifted from df_node_object_geolocation
                direction='nearest'  # Merge with the closest match in time
            )

            # Manually set the columns
            df_merged_node_car_object = pd.DataFrame(
                df_merged_node_car_object,
                columns=['Object_Label_Car', 'Object_Label_Node', 'Time_Car', 'Time_Node', 'Time_Node_Shifted', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
                         'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
                         'Node_Latitude', 'Node_Longitude', 'Node_Altitude',
                         'Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node'])

            df_merged_node_car_object_dict[delay_node] = df_merged_node_car_object  # Stored in dictionary, keyed by delay
            eucl_dist_df = calculate_euclidean_distances(wgs84, df_merged_node_car_object)
            eucl_dist_dict[delay_node] = eucl_dist_df  # Stored in dictionary, keyed by delay

            error_metrics_df = get_error_metrics(eucl_dist_df)
            error_metrics_dict[delay_node] = error_metrics_df

    relative_error_dict = getRelativeError(eucl_dist_dict)
    relative_error_diff_df = calculate_difference_on_relative_error(eucl_dist_dict)


    # Plotting.
    # plotter.plot_euclidean_distances(eucl_dist_dict)
    # plotter.plot_relative_error(relative_error_dict)
    # plotter.plot_diff_df_interactive(relative_error_diff_df)
    print('Done!')

if __name__ == '__main__':
    main()