"""
Scenario 3, Experiment 3: sensitivity of car/node object-geolocation agreement
to a synthetic rotation error injected into the node LiDAR->GPS transform.

Merged from the three near-identical originals car_node_scenario_3_exp_3_rotation_{x,y,z}.py,
which differed only in which axis the rotation was applied about. Run with
`--axis x|y|z`.
"""
import argparse
import pickle
from collections import defaultdict

import numpy as np
import nvector as nv
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from wp2.geo_utils import GeoTransformer, Plotter

DATA_ROOT = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2"


def load_data():
    df_car_annotations_lidar_cs = pd.read_csv(
        f"{DATA_ROOT}/processed_csv_files/annotations/raw/car/id-755_id-904_person_1_annotations_car.csv")
    df_node_annotations = pd.read_csv(
        f"{DATA_ROOT}/processed_csv_files/annotations/raw/node/id-720_id-869_person_1_annotations_node.csv")

    GPS_Car = (53.290474634336654, -9.071039198388549, 0)
    GPS_Node = (53.29048553013074, -9.070998387575434, 0)
    GPS0 = (53.28989834, -9.07136142, 67.566)
    GPSY = (53.28991859, -9.07135091, 67.552)
    GPSX = (53.28991224, -9.07143945, 67.531)

    # All compass headings from weather station spreadsheet for scenario 3.
    compass_list = [35.77, 35.89, 36.05, 36.14, 36.27, 36.82, 36.88]
    # 15 degree offset since the RT from car to node was aligned with 15-degree-rotated car point cloud.
    compass_heading = compass_list[0] - 15

    df_car_base = pd.read_csv(f"{DATA_ROOT}/SLAM/car_base_merge_onto_traj_gps.csv")
    df_name_R = pd.read_csv(
        f"{DATA_ROOT}/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_with_R_for_local_to_ecef.csv")
    df_name_t_int = pd.read_csv(
        f"{DATA_ROOT}/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_for_local_to_ecef_w_SLAM_T0-Tn_Interval.csv")
    df_R_t_interval = pd.merge(df_name_R, df_name_t_int, on='Base_Point')

    return df_car_annotations_lidar_cs, df_node_annotations, df_R_t_interval, GPS_Car, GPS_Node, GPS0, GPSY, GPSX, compass_heading


def preprocess_vehicle_data(df_car_annotations_lidar_cs, GPS_Car, delay_car):
    df_car_annotations_with_vehicle_location = df_car_annotations_lidar_cs
    df_car_annotations_with_vehicle_location['Base Longitude'] = GPS_Car[1]
    df_car_annotations_with_vehicle_location['Base Latitude'] = GPS_Car[0]
    df_car_annotations_with_vehicle_location['Base Altitude'] = GPS_Car[2]

    df_car_annotations_with_vehicle_location = df_car_annotations_with_vehicle_location.drop(columns=['Time'])

    time_constant_shift = 500
    shift_time_index = int((time_constant_shift / 100) - (delay_car / 100))

    df_car_annotations_with_vehicle_location['Shifted_Time'] = df_car_annotations_with_vehicle_location['Time_Short'].shift(-shift_time_index)
    df_car_annotations_with_vehicle_location['Shifted_Time'] = df_car_annotations_with_vehicle_location['Shifted_Time'].fillna(0)

    df_car_annotations_with_vehicle_location.insert(
        df_car_annotations_with_vehicle_location.columns.get_loc('Time_Short') + 1, 'Shifted_Time',
        df_car_annotations_with_vehicle_location.pop('Shifted_Time'))

    return df_car_annotations_with_vehicle_location


def preprocess_node_data(geo_transformer, df_node_annotations, delay, compass_heading):
    df_node_annotations = df_node_annotations.drop(columns=['File_Name', 'Time'])
    df_node_annotations = df_node_annotations.rename(columns={'Time_Short': 'Time'})

    # Already aligned, so no constant time shift needed (in ms).
    time_constant_shift = 0
    shift_time_index = int((time_constant_shift / 100) + (delay / 100))

    df_node_annotations['Shifted_Time'] = df_node_annotations['Time'].shift(shift_time_index)
    df_node_annotations['Shifted_Time'] = df_node_annotations['Shifted_Time'].fillna(0)

    df_node_annotations.insert(df_node_annotations.columns.get_loc('Time') + 1, 'Shifted_Time', df_node_annotations.pop('Shifted_Time'))
    return df_node_annotations


def get_axis_rotation(axis, theta):
    theta_rad = np.deg2rad(theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"Unknown axis '{axis}', expected 'x', 'y' or 'z'")


def geo_localize_car_objects(geo_transformer, df_car_annotations_with_vehicle_location, df_R_t_interval):
    df_car_annotations_with_vehicle_location['Diff'] = (
        df_car_annotations_with_vehicle_location['Time_Short'] - df_car_annotations_with_vehicle_location['Shifted_Time'])

    df_car_annotations_with_vehicle_location = pd.DataFrame(
        df_car_annotations_with_vehicle_location,
        columns=['File_Name', 'Label', 'Shifted_Time', 'Time_Short', 'Diff', 'X_Center', 'Y_Center', 'Z_Center',
                 'Length', 'Width', 'Height', 'Rx', 'Ry', 'Rz', 'Base Latitude', 'Base Longitude', 'Base Altitude']
    )

    df_car_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Car', 'Time_Car', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
        'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
        'Object_X_ECEF_Car', 'Object_Y_ECEF_Car', 'Object_Z_ECEF_Car'
    ])

    for index, row in df_car_annotations_with_vehicle_location.iterrows():
        shifted_time = row['Shifted_Time']
        row['Time'] = row['Time_Short']
        if shifted_time == 0:
            df_car_object_geolocation = pd.concat([df_car_object_geolocation, pd.DataFrame({
                'Object_Label_Car': [row['Label']],
                'Time_Car': [row['Time']],
                'Time_Car_Shifted': [row['Shifted_Time']],
                'Car_Latitude': [row['Base Latitude']],
                'Car_Longitude': [row['Base Longitude']],
                'Car_Altitude': [0],
                'Object_Latitude_Car': [row['Base Latitude']],
                'Object_Longitude_Car': [row['Base Longitude']],
                'Object_Altitude_Car': [0],
                'Object_X_ECEF_Car': [0],
                'Object_Y_ECEF_Car': [0],
                'Object_Z_ECEF_Car': [0],
                'X': [0], 'Y': [0], 'Z': [0]
            })], ignore_index=True)
        else:
            theta_lidar_to_local = np.deg2rad(-15)
            R_LidarToLocal = [[np.cos(theta_lidar_to_local), -np.sin(theta_lidar_to_local), 0],
                              [np.sin(theta_lidar_to_local), np.cos(theta_lidar_to_local), 0],
                              [0, 0, 1]]

            R_LocalToGlobal = geo_transformer.getLidarToLocalCS_Rotation(
                df_R_t_interval, row['Time'], geo_transformer, hard_values=False)

            matching_row = df_car_annotations_with_vehicle_location[df_car_annotations_with_vehicle_location['Time_Short'] == shifted_time].iloc[0]
            x_off, y_off, z_off = matching_row['X_Center'], matching_row['Y_Center'], matching_row['Z_Center']
            GPS_base_curr = (matching_row['Base Latitude'], matching_row['Base Longitude'], matching_row['Base Altitude'])

            point_lidar_cs = np.array([x_off, y_off, z_off])
            point_local_cs = np.dot(R_LidarToLocal, point_lidar_cs)
            point_local_cs_left_neg = np.array([-point_local_cs[0], point_local_cs[1], point_local_cs[2]])

            ECEF_base_curr = geo_transformer.gps_to_ecef(*GPS_base_curr)
            point_global_cs_ecef = geo_transformer.lidar_to_ecef(point_local_cs_left_neg, ECEF_base_curr, R_LocalToGlobal)
            gps_point = geo_transformer.ecef_to_gps(*point_global_cs_ecef)

            df_car_object_geolocation = pd.concat([df_car_object_geolocation, pd.DataFrame({
                'Object_Label_Car': [row['Label']],
                'Time_Car': [row['Shifted_Time']],
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


def geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading, axis, value, delay):
    df_node_object_geolocation = pd.DataFrame(columns=[
        'Object_Label_Node', 'Rotation_Offset', 'Time_Node', 'Time_Node_Shifted', 'Node_Latitude', 'Node_Longitude', 'Node_Altitude',
        'Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node',
        'Object_X_ECEF_Node', 'Object_Y_ECEF_Node', 'Object_Z_ECEF_Node',
        'X', 'Y', 'Z'
    ])

    for index, row in df_node_annotations.iterrows():
        shifted_time = row['Shifted_Time']

        if shifted_time == 0:
            df_node_object_geolocation = pd.concat([df_node_object_geolocation, pd.DataFrame({
                'Object_Label_Node': [row['Label']],
                'Rotation_Offset': [0],
                'Time_Node': [row['Time']],
                'Time_Node_Shifted': [row['Shifted_Time']],
                'Node_Latitude': [row['Base Latitude']],
                'Node_Longitude': [row['Base Longitude']],
                'Node_Altitude': [0],
                'Object_Latitude_Node': [row['Base Latitude']],
                'Object_Longitude_Node': [row['Base Longitude']],
                'Object_Altitude_Node': [0],
                'Object_X_ECEF_Node': [0],
                'Object_Y_ECEF_Node': [0],
                'Object_Z_ECEF_Node': [0],
                'X': [0], 'Y': [0], 'Z': [0]
            })], ignore_index=True)
        else:
            matching_row = df_node_annotations[df_node_annotations['Time'] == shifted_time].iloc[0]
            x_off, y_off, z_off = matching_row['X_Center'], matching_row['Y_Center'], matching_row['Z_Center']
            GPS_base_curr = (matching_row['Base Latitude'], matching_row['Base Longitude'])

            lidar_point = np.array([x_off, y_off, z_off])

            rot_mat = get_axis_rotation(axis, value)
            lidar_rot_point = np.dot(rot_mat, lidar_point)
            gps_point = geo_transformer.node_geolocate_object(GPS_base_curr, compass_heading, lidar_rot_point)
            ecef_point = geo_transformer.gps_to_ecef(*gps_point)

            df_node_object_geolocation = pd.concat([df_node_object_geolocation, pd.DataFrame({
                'Object_Label_Node': [row['Label']],
                'Rotation_Offset': [value],
                'Time_Node': [row['Time']],
                'Time_Node_Shifted': [row['Shifted_Time']],
                'Node_Latitude': [GPS_base_curr[0]],
                'Node_Longitude': [GPS_base_curr[1]],
                'Node_Altitude': [0],
                'Object_Latitude_Node': [gps_point[1]],
                'Object_Longitude_Node': [gps_point[0]],
                'Object_Altitude_Node': [0],
                'Object_X_ECEF_Node': [ecef_point[0]],
                'Object_Y_ECEF_Node': [ecef_point[1]],
                'Object_Z_ECEF_Node': [ecef_point[2]],
                'X': [lidar_rot_point[0]],
                'Y': [lidar_rot_point[1]],
                'Z': [lidar_rot_point[2]]
            })], ignore_index=True)

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

            dist_eucl_car_base_to_node_base = (point_car_gps.to_ecef_vector() - point_node_gps.to_ecef_vector()).length
            dist_eucl_car_obj_to_node_obj = (point_obj_car_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()).length
            dist_eucl_car_base_to_car_obj = (point_car_gps.to_ecef_vector() - point_obj_car_gps.to_ecef_vector()).length
            dist_eucl_node_base_to_node_obj = (point_node_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()).length
            dist_eucl_car_base_to_node_obj = (point_car_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()).length
            dist_eucl_node_base_to_car_obj = (point_node_gps.to_ecef_vector() - point_obj_car_gps.to_ecef_vector()).length
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


def calculate_difference_on_relative_error(eucl_dist_dict):
    # Delay 0 is the baseline; every other delay's error is reported relative to it.
    diff_df = pd.DataFrame(columns=['Time_Car', 'Label_Car', 'Label_Node', 'node_base_to_node_obj'] +
                                   ['Diff_' + str(0) + '-' + str(delay) for delay in sorted(eucl_dist_dict.keys()) if delay != 0])

    sorted_delays = sorted(eucl_dist_dict.keys())
    if 0 not in sorted_delays:
        raise ValueError("Key 0 (baseline) must be present in `eucl_dist_dict`.")

    for label_node in eucl_dist_dict[0]['Label_Node'].unique():
        for time_car in eucl_dist_dict[0]['Time_Car'].unique():
            baseline_row = eucl_dist_dict[0][(eucl_dist_dict[0]['Time_Car'] == time_car) &
                                             (eucl_dist_dict[0]['Label_Node'] == label_node)]
            if not baseline_row.empty:
                baseline_error = baseline_row['car_obj_to_node_obj'].values[0]
                diff_data = {
                    'Time_Car': time_car,
                    'Label_Car': baseline_row['Label_Car'].values[0],
                    'Label_Node': label_node,
                    'node_base_to_node_obj': baseline_row['node_base_to_node_obj']
                }

                for delay in sorted_delays:
                    if delay != 0:
                        current_row = eucl_dist_dict[delay][(eucl_dist_dict[delay]['Time_Car'] == time_car) &
                                                            (eucl_dist_dict[delay]['Label_Node'] == label_node)]
                        if not current_row.empty:
                            current_error = current_row['car_obj_to_node_obj'].values[0]
                            current_node_base_to_node_obj = current_row['node_base_to_node_obj'].values[0]
                            diff_data['Diff_' + str(0) + '-' + str(delay)] = abs(current_error - baseline_error)
                            diff_data['node_base_to_node_obj'] = current_node_base_to_node_obj
                        else:
                            diff_data['Diff_' + str(0) + '-' + str(delay)] = None
                            diff_data['node_base_to_node_obj'] = None

                diff_df = pd.concat([diff_df, pd.DataFrame([diff_data])], ignore_index=True)
    return diff_df


def _extract_filtered_delays(diff_df, diff_columns, delay_range=(0, 5)):
    delays = []
    for col in diff_columns:
        try:
            delays.append(float(col.split('_')[1].split('-')[1]))
        except ValueError:
            pass
    filtered_delays = [d for d in delays if delay_range[0] <= d <= delay_range[1]]
    filtered_indices = [i for i, d in enumerate(delays) if delay_range[0] <= d <= delay_range[1]]
    return filtered_delays, filtered_indices


def plot_diff_df_interactive_with_averages(diff_df, axis):
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m']

    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    filtered_delays, filtered_indices = _extract_filtered_delays(diff_df, diff_columns)

    fig = go.Figure()

    for idx, row in diff_df.iterrows():
        diff_values = [row[col] for col in diff_columns]
        filtered_diff_values = [diff_values[i] for i in filtered_indices]
        time_car = row['Time_Car'] if 'Time_Car' in row else 'N/A'
        node_base_to_node_obj = row['node_base_to_node_obj'] if 'node_base_to_node_obj' in row else 'N/A'

        fig.add_trace(go.Scatter(
            x=filtered_delays, y=filtered_diff_values, mode='lines+markers',
            name=f'Row: {idx}, Time: {time_car}, Node To Obj Dist: {node_base_to_node_obj}',
            visible='legendonly'
        ))

    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])
        filtered_diff_values = [diff_values[i] for i in filtered_indices]
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(filtered_diff_values)

    for range_label, values in range_averages.items():
        avg_values = np.nanmean(values, axis=0)
        fig.add_trace(go.Scatter(x=filtered_delays, y=avg_values, mode='lines+markers', name=f'Average ({range_label})', visible=True))

    fig.update_layout(
        title='Difference in Car-to-Node Object Distance (Averaged and Individual Rows)',
        xaxis_title=f'{axis.upper()}_Rotation (degrees)',
        yaxis_title='Error/Relative Difference (m)',
        legend_title='Rows and Averages (Click to toggle)',
        hovermode='x', template='plotly',
        xaxis=dict(tickmode='array', tickvals=filtered_delays, showgrid=True, tickangle=90),
        yaxis=dict(showgrid=True),
    )
    fig.show()


def plot_diff_df_with_averages(diff_df, axis):
    distance_ranges = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18)]
    range_labels = ['8-10m', '10-12m', '12-14m', '14-16m', '16-18m']

    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    filtered_delays, filtered_indices = _extract_filtered_delays(diff_df, diff_columns)

    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])
        filtered_diff_values = [diff_values[i] for i in filtered_indices]
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(filtered_diff_values)

    for range_label, values in range_averages.items():
        range_averages[range_label] = np.nanmean(values, axis=0)

    plt.figure(figsize=(10, 6))
    for range_label in range_labels:
        if range_label in range_averages:
            plt.plot(filtered_delays, range_averages[range_label], label=range_label, marker='o')

    plt.title(f'Impact of {axis.upper()} Axis Rotation Error on Spatial Uncertainty', fontsize=16)
    plt.xlabel(f'{axis.upper()} Axis Rotation (degrees)', fontsize=16)
    plt.ylabel('Euclidean Distance Between Vehicle Detection and Node Detection (m)', fontsize=16)
    plt.xticks(filtered_delays, rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Distance Ranges', fontsize=14)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()


def main(axis):
    geo_transformer = GeoTransformer()
    plotter = Plotter()
    wgs84 = nv.FrameE(name='WGS84')
    df_car_annotations_lidar_cs, df_node_annotations_orig, df_R_t_interval, GPS_Car, GPS_Node, GPS0, GPSY, GPSX, compass_heading = load_data()

    theta_list = np.round(np.arange(-5, 5, 0.1), 2)

    delay_car = 0
    delay_node = 0
    eucl_dist_dict = {}
    df_merged_node_car_object_dict = {}

    for theta in theta_list:
        df_car_annotations_with_vehicle_location = preprocess_vehicle_data(df_car_annotations_lidar_cs, GPS_Car, delay_car)
        df_node_annotations = preprocess_node_data(geo_transformer, df_node_annotations_orig, delay_node, compass_heading)
        df_car_object_geolocation = geo_localize_car_objects(geo_transformer, df_car_annotations_with_vehicle_location, df_R_t_interval)
        df_node_object_geolocation = geo_localize_node_objects(geo_transformer, df_node_annotations, compass_heading, axis, theta, delay_node)

        df_merged_node_car_object = pd.merge_asof(
            df_car_object_geolocation.sort_index(),
            df_node_object_geolocation.sort_index(),
            left_index=True, right_index=True, direction='nearest'
        )
        df_merged_node_car_object = pd.DataFrame(
            df_merged_node_car_object,
            columns=['Object_Label_Car', 'Object_Label_Node', 'Time_Car', 'Time_Node', 'Time_Node_Shifted', 'Car_Latitude', 'Car_Longitude', 'Car_Altitude',
                     'Object_Latitude_Car', 'Object_Longitude_Car', 'Object_Altitude_Car',
                     'Node_Latitude', 'Node_Longitude', 'Node_Altitude',
                     'Object_Latitude_Node', 'Object_Longitude_Node', 'Object_Altitude_Node'])

        df_merged_node_car_object_dict[theta] = df_merged_node_car_object
        eucl_dist_dict[theta] = calculate_euclidean_distances(wgs84, df_merged_node_car_object)

    relative_error_diff_df = calculate_difference_on_relative_error(eucl_dist_dict)

    with open(f'pickles/scenario_3_exp_3_{axis}.pkl', 'wb') as f:
        pickle.dump({
            'eucl_dist_dict': eucl_dist_dict,
            'relative_error_diff_df': relative_error_diff_df
        }, f)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', choices=['x', 'y', 'z'], required=True, help='Rotation axis to sweep')
    args = parser.parse_args()
    main(args.axis)
