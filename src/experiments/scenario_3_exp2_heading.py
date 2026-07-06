import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from wp2.geo_utils import GeoTransformer, Plotter
import nvector as nv
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import pickle

def load_data():
    # Important Data.
    df_car_annotations_lidar_cs = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-755_id-904_person_1_annotations_car.csv")
    df_node_annotations = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_person_1_annotations_node.csv')

    GPS_Car = (53.290474634336654, -9.071039198388549, 0)
    GPS_Node = (53.29048553013074, -9.070998387575434, 0)
    GPS0 = (53.28989834, -9.07136142, 67.566)
    GPSY = (53.28991859, -9.07135091, 67.552)
    GPSX = (53.28991224, -9.07143945, 67.531)


    compass_list = [35.77, 35.89, 36.05, 36.14, 36.27, 36.82, 36.88] # All compass heading from weather station spreadsheet for scenario 3.
    compass_heading = compass_list[0] - 15 # 15 degrees offset since the RT from car to node was aligned with 15 degrees rotated car point cloud.

    # Supplementary Data.
    df_car_base = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/SLAM/car_base_merge_onto_traj_gps.csv')
    df_name_R = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_with_R_for_local_to_ecef.csv')
    df_name_t_int = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_for_local_to_ecef_w_SLAM_T0-Tn_Interval.csv')
    # Merged R matrix, and time interval from df_name_R and df_name_t_int.
    df_R_t_interval = pd.merge(df_name_R, df_name_t_int, on='Base_Point')

    return df_car_annotations_lidar_cs, df_node_annotations, df_R_t_interval, GPS_Car, GPS_Node, GPS0, GPSY, GPSX, compass_heading

def preprocess_vehicle_data(df_car_annotations_lidar_cs, GPS_Car, delay_car):
    df_car_annotations_with_vehicle_location = df_car_annotations_lidar_cs
    df_car_annotations_with_vehicle_location['Base Longitude'] = GPS_Car[1]
    df_car_annotations_with_vehicle_location['Base Latitude'] = GPS_Car[0]
    df_car_annotations_with_vehicle_location['Base Altitude'] = GPS_Car[2]

    df_car_annotations_with_vehicle_location = df_car_annotations_with_vehicle_location.drop(columns=['Time'])

    # time_constant_shift = 800
    time_constant_shift = 500
    # Should you minus or add the delay?
    shift_time_index = int((time_constant_shift / 100) - (delay_car / 100))

    # Shift the time on df_trajectory_global_cs by `shift_time_index` to get spatial alignment.
    df_car_annotations_with_vehicle_location['Shifted_Time'] = df_car_annotations_with_vehicle_location['Time_Short'].shift(-shift_time_index)
    df_car_annotations_with_vehicle_location['Shifted_Time'] = df_car_annotations_with_vehicle_location['Shifted_Time'].fillna(0)

    df_car_annotations_with_vehicle_location.insert(df_car_annotations_with_vehicle_location.columns.get_loc('Time_Short') + 1, 'Shifted_Time',
                                   df_car_annotations_with_vehicle_location.pop('Shifted_Time'))


    return df_car_annotations_with_vehicle_location

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
    return df_node_annotations

def geo_localize_car_objects(geo_transformer, df_car_annotations_with_vehicle_location, df_R_t_interval):
    df_car_annotations_with_vehicle_location['Diff'] = (df_car_annotations_with_vehicle_location['Time_Short'] - df_car_annotations_with_vehicle_location['Shifted_Time'])

    # Set the columns
    df_car_annotations_with_vehicle_location = pd.DataFrame(
        df_car_annotations_with_vehicle_location,
        columns=['File_Name', 'Label', 'Shifted_Time', 'Time_Short', 'Diff', 'X_Center', 'Y_Center', 'Z_Center',
                 'Length', 'Width', 'Height', 'Rx', 'Ry', 'Rz', 'Base Latitude', 'Base Longitude', 'Base Altitude']
    )

    #### Recursive Duplicate Handling ####
    # Keep running until no duplicates exist in the 'Shifted_Time' column.
    # Removed from here. Check car_node_scenario_2_exp_1_delay.py for the recursive duplicate handling.
    #### End of Recursive Duplicate Handling ####

    # Create output DataFrame for car object geolocation
    df_car_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Car', 'Time_Car', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
        'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
        'Object_X_ECEF_Car', 'Object_Y_ECEF_Car', 'Object_Z_ECEF_Car'
    ])

    # Iterate through rows in the updated DataFrame for geolocation
    for index, row in df_car_annotations_with_vehicle_location.iterrows():
        shifted_time = row['Shifted_Time']
        row['Time'] = row['Time_Short']
        if shifted_time == 0:
            # continue
            df_car_object_geolocation = pd.concat([df_car_object_geolocation, pd.DataFrame({
                'Object_Label_Car': [row['Label']],
                'Time_Car': [row['Time']],
                'Time_Car_Shifted': [row['Shifted_Time']],
                # This is the shifted time, i.e, the time with the delay of processing.
                'Car_Latitude': [row['Base Latitude']],
                'Car_Longitude': [row['Base Longitude']],
                'Car_Altitude': [0],
                'Object_Latitude_Car': [row['Base Latitude']],
                'Object_Longitude_Car': [row['Base Longitude']],
                'Object_Altitude_Car': [0],
                'Object_X_ECEF_Car': [0],
                'Object_Y_ECEF_Car': [0],
                'Object_Z_ECEF_Car': [0],
                'X': [0],
                'Y': [0],
                'Z': [0]
            })], ignore_index=True)
            # print(f"200 Car_0%_Heading_Error_T{index} {row['Base Longitude']} {row['Base Latitude']} 0 1")

        else:
            # Rotation matrix to convert from Lidar to local coordinate system
            theta_lidar_to_local = np.deg2rad(-15)
            R_LidarToLocal = [[np.cos(theta_lidar_to_local), -np.sin(theta_lidar_to_local), 0],
                              [np.sin(theta_lidar_to_local), np.cos(theta_lidar_to_local), 0],
                              [0, 0, 1]]

            # Get the rotation matrix from local to global
            R_LocalToGlobal = geo_transformer.getLidarToLocalCS_Rotation(
                df_R_t_interval, row['Time'], geo_transformer, hard_values=False)

            matching_row = df_car_annotations_with_vehicle_location[df_car_annotations_with_vehicle_location['Time_Short'] == shifted_time].iloc[0]
            x_off, y_off, z_off = matching_row['X_Center'], matching_row['Y_Center'], matching_row['Z_Center']
            GPS_base_curr = (matching_row['Base Latitude'], matching_row['Base Longitude'], matching_row['Base Altitude'])

            point_lidar_cs = np.array([x_off, y_off, z_off])
            point_local_cs = np.dot(R_LidarToLocal, point_lidar_cs)
            point_local_cs_left_neg = np.array([-point_local_cs[0], point_local_cs[1], point_local_cs[2]])

            # Convert GPS Coordinates
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
            # print(f"200 Car_0%_Heading_Error_T{index} {gps_point[0]} {gps_point[1]} {gps_point[2]} 1")
    return df_car_object_geolocation

def geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading, compass_heading_pc_err, delay):
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
            print(f"200 NODE_{compass_heading_pc_err}_T{index} {row['Base Longitude']} {row['Base Latitude']} 0 1")
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
            print(f"2 Node_{compass_heading_pc_err}_T{index} {gps_point[0]} {gps_point[1]} 0 1")


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
    relative_error_dict = {}
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

def calculate_difference_on_relative_error(eucl_dist_dict):
    # Ensure the baseline (0% heading_pc_error) exists in the dictionary
    if 0.0 not in eucl_dist_dict:
        raise ValueError("Key 0 (baseline) must be present in `eucl_dist_dict`.")

    # Get a sorted list of heading % errors
    sorted_heading = sorted(eucl_dist_dict.keys())

    # Initialize the output DataFrame with appropriate columns, ensuring 0.0 is included.
    diff_columns = ['Diff_' + str(0) + '-' + str(key) for key in sorted_heading]
    diff_df = pd.DataFrame(columns=['Time_Car', 'Label_Car', 'Label_Node', 'node_base_to_node_obj'] + diff_columns)

    # Iterate over all unique Label_Node and Time_Car combinations in the baseline data (key 0)
    for label_node in eucl_dist_dict[0.0]['Label_Node'].unique():
        for time_car in eucl_dist_dict[0.0]['Time_Car'].unique():

            # Row for the given Label_Node and Time_Car in the baseline (0% heading error)
            baseline_row = eucl_dist_dict[0.0][(eucl_dist_dict[0.0]['Time_Car'] == time_car) &
                                               (eucl_dist_dict[0.0]['Label_Node'] == label_node)]

            # Check if the baseline data exists for this combination
            if not baseline_row.empty:
                baseline_error = baseline_row['car_obj_to_node_obj'].values[0]

                # Initialize the difference dictionary for this combination
                diff_data = {
                    'Time_Car': time_car,
                    'Label_Car': baseline_row['Label_Car'].values[0],
                    'Label_Node': label_node,
                    'node_base_to_node_obj': baseline_row['node_base_to_node_obj'].values[0]
                }

                # Iterate over all heading % errors, including 0.0
                for heading_pc_error in sorted_heading:
                    current_row = eucl_dist_dict[heading_pc_error][(eucl_dist_dict[heading_pc_error]['Time_Car'] == time_car) &
                                                                   (eucl_dist_dict[heading_pc_error]['Label_Node'] == label_node)]

                    if not current_row.empty:
                        current_error = current_row['car_obj_to_node_obj'].values[0]
                        current_node_base_to_node_obj = current_row['node_base_to_node_obj'].values[0]

                        # Compute the absolute difference
                        diff_data['Diff_' + str(0) + '-' + str(heading_pc_error)] = abs(current_error - baseline_error)
                        diff_data['node_base_to_node_obj'] = current_node_base_to_node_obj
                    else:
                        diff_data['Diff_' + str(0) + '-' + str(heading_pc_error)] = None
                        diff_data['node_base_to_node_obj'] = None

                # Append the computed difference data to the output DataFrame
                diff_df = pd.concat([diff_df, pd.DataFrame([diff_data])], ignore_index=True)

    return diff_df

def calculate_baseline_distances(wgs84, df_node_object_geolocation_dict):
    """Compute distances from the baseline (0% heading error) to other node objects."""
    all_rows = []

    # Extract baseline data (key=0)
    baseline_df = df_node_object_geolocation_dict[0]
    baseline_coords = baseline_df[['Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node']].to_numpy()

    # Iterate over all rows in the baseline data
    for row_index, baseline_row in enumerate(baseline_coords):
        row_data = []  # Store distances for the current row across all headings
        for heading_error, node_object_df in df_node_object_geolocation_dict.items():
            # Get the corresponding point at the current heading error
            current_row = node_object_df.iloc[row_index]
            current_coords = [current_row['Object_Latitude_Node'],
                              current_row['Object_Longitude_Node'],
                              current_row['Object_Altitude_Node']]

            # Calculate distance from baseline to current point
            baseline_geo_point = wgs84.GeoPoint(latitude=baseline_row[0], longitude=baseline_row[1], degrees=True)
            current_geo_point = wgs84.GeoPoint(latitude=current_coords[0], longitude=current_coords[1], degrees=True)
            ecef_diff = baseline_geo_point.to_ecef_vector() - current_geo_point.to_ecef_vector()
            distance = ecef_diff.length

            # Save the information for this row and heading error
            row_data.append({
                'Row_Index': row_index,
                'Time_Node': current_row['Time_Node'],
                'Heading_Error': heading_error,
                'Baseline_to_Current_Node_Obj_Distance': distance,
            })

        # Concatenate the row's data for all heading errors
        all_rows.extend(row_data)

    # Combine all rows into a dataframe
    distance_df = pd.DataFrame(all_rows)
    return distance_df

def plot_baseline_to_node_distances(distance_df):
    """Generate an interactive plot with all graphs toggled off by default."""

    # Add identifying information for legend labels
    distance_df['Row_Info'] = (
            "Row: " + distance_df['Row_Index'].astype(str) +
            ", Time: " + distance_df['Time_Node'].round(2).astype(str)
    )

    # Create a line plot
    fig = px.line(
        distance_df,
        x='Heading_Error',
        y='Baseline_to_Current_Node_Obj_Distance',
        color='Row_Info',
        labels={
            'Heading_Error': 'Heading Error (%)',
            'Baseline_to_Current_Node_Obj_Distance': 'Distance from Node Baseline to Node Obj. (m)'
        },
        title="Distance from Node Baseline to Node Detected Object"
    )

    # Update the traces: set visibility to 'legendonly' (toggle off all graphs initially)
    fig.for_each_trace(lambda trace: trace.update(visible='legendonly'))

    # Update layout for better interactivity
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        legend_title="Rows (Click to toggle)",
        xaxis=dict(title="Heading Error (%)"),
        yaxis=dict(title="Distance from Node Baseline to Node Obj. (m)"),
        plot_bgcolor="lightblue",
        legend=dict(
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=10)
        )
    )

    fig.show()

def plot_diff_df_interactive_with_averages(diff_df):
    # Define distance ranges
    distance_ranges = [
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22)
    ]
    range_labels = [
        '8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m', '14-15m',
        '15-16m', '16-17m', '17-18m', '18-19m', '19-20m', '20-21m', '21-22m'
    ]

    # Extract difference columns
    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    delays = []
    for col in diff_columns:
        try:
            delay = float(col.split('_')[1].split('-')[1])  # Extract the delay
            delays.append(delay)
        except ValueError:
            pass

    # Initialize the figure
    fig = go.Figure()

    # Plot individual rows
    for idx, row in diff_df.iterrows():
        diff_values = [row[col] for col in diff_columns]
        time_car = row['Time_Car'] if 'Time_Car' in row else 'N/A'
        node_base_to_node_obj = row['node_base_to_node_obj'] if 'node_base_to_node_obj' in row else 'N/A'

        fig.add_trace(go.Scatter(
            x=delays,
            y=diff_values,
            mode='lines+markers',
            name=f'Row: {idx}, Time: {time_car}, Node To Obj Dist: {node_base_to_node_obj}',
            visible='legendonly'
        ))

    # Calculate averages for defined distance ranges
    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])

        # Determine the range label for this row
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(diff_values)

    # Compute average differences and plot for each range
    for range_label, values in range_averages.items():
        avg_values = np.nanmean(values, axis=0)  # Compute the mean across rows
        fig.add_trace(go.Scatter(
            x=delays,
            y=avg_values,
            mode='lines+markers',
            name=f'Average ({range_label})',
            visible=True  # Ensure averages are always visible initially
        ))

    # Update layout
    fig.update_layout(
        title='Difference in Car-to-Node Object Distance (Averaged and Individual Rows)',
        xaxis_title='Heading Error (%)',
        yaxis_title='Difference in Car-to-Node Object Distance (m)',
        legend_title='Rows and Averages (Click to toggle)',
        hovermode='x',
        template='plotly',
        xaxis=dict(tickmode='array', tickvals=delays, showgrid=True),
        yaxis=dict(showgrid=True),
    )

    # Show the plot
    fig.show()

def plot_diff_df_with_averages(diff_df):
    # Define distance ranges and corresponding labels
    distance_ranges = [(8, 9), (9, 10), (10, 11)]
    range_labels = ['8-9m', '9-10m', '10-11m']

    # Extract columns that start with "Diff_" for differences data
    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]

    # Extract delays from column names
    delays = []
    for col in diff_columns:
        try:
            delay = float(col.split('_')[1].split('-')[1])  # Extract delay from column name
            delays.append(delay)
        except ValueError:
            pass

    # Filter delays to keep only a specific range and find their indices
    filtered_delays = [delay for delay in delays if 0 <= delay <= 15]
    filtered_indices = [i for i, delay in enumerate(delays) if 0 <= delay <= 15]

    # Calculate averages for rows within each distance range
    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])

        # Filter the difference values based on the filtered delay indices
        filtered_diff_values = [diff_values[i] for i in filtered_indices]

        # Assign the values to the appropriate distance range
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(filtered_diff_values)

    # Compute average differences and store them
    for range_label, values in range_averages.items():
        range_averages[range_label] = np.nanmean(values, axis=0)  # Handle NaN values correctly

    # Plot the averages for each distance range
    plt.figure(figsize=(10, 6))
    for range_label in range_labels:
        if range_label in range_averages:
            plt.plot(
                filtered_delays,
                range_averages[range_label],
                label=range_label,
                marker='o'  # Use markers to distinguish data points
            )

    # Customize the plot for clarity and presentation
    plt.title('Impact of GPS Heading Error on Spatial Uncertainity', fontsize=16)
    plt.xlabel('Heading Error ($^\circ$)', fontsize=14)
    plt.ylabel('Euclidean Distance Between Vehicle Detection and Node Detection (m)', fontsize=14)
    plt.xticks(filtered_delays, rotation=0, fontsize=12)  # Customize X-axis labels
    plt.yticks(fontsize=12)
    plt.legend(title='Distance Ranges', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust axes if necessary
    plt.tight_layout()
    plt.show()

def main():
    geo_transformer = GeoTransformer()
    plotter = Plotter()
    wgs84 = nv.FrameE(name='WGS84')
    df_car_annotations_lidar_cs, df_node_annotations_orig, df_R_t_interval, GPS_Car, GPS_Node, GPS0, GPSY, GPSX, compass_heading = load_data()

    # heading_node = [0]
    heading_node = [compass_heading + i * 0.1 for i in range(-20, 21)]
    heading_errors = [round(i * 0.1, 2) for i in range(-20, 21)]

    # heading_node = [compass_heading + i * 0.1 for i in range(-2, 3)]
    # heading_errors = [round(i * 0.1, 2) for i in range(-2, 3)]


    delay_car = 0
    delay_node = 0
    eucl_dist_dict = {}
    df_merged_node_car_object_dict = {}
    df_node_object_geolocation_dict = {}
    df_car_object_geolocation_dict = {}


    for heading_pc_err, heading in zip(heading_errors, heading_node):
        df_car_annotations_with_vehicle_location = preprocess_vehicle_data(df_car_annotations_lidar_cs, GPS_Car, delay_car)
        df_node_annotations = preprocess_node_data(geo_transformer, df_node_annotations_orig, delay_node, heading)

        df_car_object_geolocation = geo_localize_car_objects(geo_transformer, df_car_annotations_with_vehicle_location, df_R_t_interval)
        print('\n')
        df_node_object_geolocation = geo_localize_node_objects(geo_transformer, df_node_annotations, heading, heading_pc_err, delay_node)

        df_car_object_geolocation_dict[heading_pc_err] = df_car_object_geolocation_dict
        df_node_object_geolocation_dict[heading_pc_err] = df_node_object_geolocation

        df_merged_node_car_object = pd.merge_asof(
            df_car_object_geolocation.sort_index(),
            df_node_object_geolocation.sort_index(),  # Sort by index
            left_index=True,  # Merge on index from df_car_object_geolocation
            right_index=True,  # Merge on index from df_node_object_geolocation
            direction='nearest'  # Merge with the closest match in time
        )

        # Manually set the columns
        df_merged_node_car_object = pd.DataFrame(
            df_merged_node_car_object,
            columns=['Object_Label_Car', 'Object_Label_Node', 'Time_Car', 'Time_Node', 'Time_Node_Shifted', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
                     'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
                     'Node_Latitude', 'Node_Longitude', 'Node_Altitude',
                     'Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node'])

        df_merged_node_car_object_dict[heading_pc_err] = df_merged_node_car_object
        eucl_dist_df = calculate_euclidean_distances(wgs84, df_merged_node_car_object)
        eucl_dist_dict[heading_pc_err] = eucl_dist_df

    relative_error_diff_df = calculate_difference_on_relative_error(eucl_dist_dict)
    relative_error_dict = getRelativeError(eucl_dist_dict)

    with open('pickles/scenario_3_exp_2_3.pkl', 'wb') as f:
        pickle.dump({
            'eucl_dist_dict': eucl_dist_dict,
            'relative_error_dict': relative_error_dict,
            'relative_error_diff_df': relative_error_diff_df
        }, f)

    # with open('pickles/scenario_3_exp_2_2.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # eucl_dist_dict = data['eucl_dist_dict']
    # relative_error_dict = data['relative_error_dict']
    # relative_error_diff_df = data['relative_error_diff_df']


    # Plotting.
    # plotter.plot_euclidean_distances(eucl_dist_dict)
    # plotter.plot_relative_error(relative_error_dict)
    # plot_diff_df_interactive_with_averages(relative_error_diff_df)
    # plot_diff_df_with_averages(relative_error_diff_df)

    # print('Done!')

if __name__ == '__main__':
    main()