from pyproj import CRS, Transformer
import numpy as np
import pandas as pd
import math
import geopy.distance
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GeoTransformer:
    def __init__(self):
        self.wgs84 = CRS("EPSG:4326")
        self.ecef = CRS("EPSG:4978")

    def gps_to_ecef(self, lat, lon, alt):
        wgs84_to_ecef = Transformer.from_crs(self.wgs84, self.ecef)
        x, y, z = wgs84_to_ecef.transform(lat, lon, alt)
        return np.array([x, y, z])

    def ecef_to_gps(self, x, y, z):
        ecef_to_wgs84 = Transformer.from_crs(self.ecef, self.wgs84)
        point_lat, point_lon, point_alt = ecef_to_wgs84.transform(x, y, z)
        return point_lon, point_lat, point_alt

    @staticmethod
    def normalize(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def lidar_to_ecef(lidar_point, ECEF0, R):
        ECEF_point = R @ lidar_point + ECEF0
        return ECEF_point

    def get_rotation(self, ECEF0, ECEFX, ECEFY):
        V_Y = ECEFY - ECEF0
        V_X = ECEFX - ECEF0

        U_Y = self.normalize(V_Y)
        U_X = self.normalize(V_X)
        U_Z = self.normalize(np.cross(U_Y, U_X))
        U_X = self.normalize(np.cross(U_Z, U_Y))
        R = np.vstack((U_X, U_Y, U_Z)).T

        return R

    @staticmethod
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
        R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                            [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                            [0, 0, 1]]
        df_rotated[['X', 'Y', 'Z']] = df_rotated.apply(
            lambda row: pd.Series(np.dot(R_lidar_to_local, row[['X', 'Y', 'Z']].values)), axis=1)
        return df_rotated

    @staticmethod
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
            df_gps_list.append(pd.DataFrame(
                {'Colour_ID': [0], 'Name': [row['Time']], 'Longitude': [gps_point[0]], 'Latitude': [gps_point[1]],
                 'Altitude': [gps_point[2]],
                 'Show': [1]}))

            # print(row['Time'], df.loc[index, 'X'], df.loc[index, 'Y'], df.loc[index, 'Z'])
            df_traj_gps_list.append(pd.DataFrame({'Time': [row['Time']], 'Rx(Roll)': [df.loc[index, 'Rx(Roll)']],
                                                  'Ry(Pitch)': [df.loc[index, 'Ry(Pitch)']],
                                                  'Rz(Yaw)': [df.loc[index, 'Rz(Yaw)']], 'X': [df.loc[index, 'X']],
                                                  'Y': [df.loc[index, 'Y']], 'Z': [df.loc[index, 'Z']],
                                                  'Base Longitude': [gps_point[0]], 'Base Latitude': [gps_point[1]],
                                                  'Base Altitude': [gps_point[2]]}))

        df_gps = pd.concat(df_gps_list, ignore_index=True)
        df_traj_gps = pd.concat(df_traj_gps_list, ignore_index=True)
        df_merged = pd.concat([df_gps, df_car_base], ignore_index=True)
        return df_merged, df_traj_gps

    @staticmethod
    def getLidarToLocalCS_Rotation(df_R_t_interval, time_n, geoTransformer, hard_values):
        # Old. Before sc3.
        # if time_n < df_R_t_interval['Time_Start'].min():
        #     hard_values = True
        if time_n < df_R_t_interval['Time_Start'].min() or time_n > df_R_t_interval['Time_Start'].max():
            hard_values = True
        else:
            hard_values = False
        if not hard_values:
            closest_time_start_index = (df_R_t_interval['Time_Start'] - time_n).abs().idxmin()
            closest_time_start = df_R_t_interval.loc[closest_time_start_index, 'Time_Start']
            # Check if closest_time_start is greater than time_1
            if closest_time_start > time_n:
                # If it is, find the previous index
                closest_time_start_index -= 1
                # Get the new closest_time_start
                closest_time_start = df_R_t_interval.loc[closest_time_start_index, 'Time_Start']
            R_str = df_R_t_interval.loc[closest_time_start_index, 'R']
            R_str = R_str.strip('[]')  # Remove the square brackets from the string
            R = np.fromstring(R_str, sep=',')  # Convert the string back to a numpy array
            R = R.reshape(3, 3)

        if hard_values:  # For testing purposes. Set True to use the hardcoded values. Car_base_1 = Origin, Car_base_2 = Positive Y direction, Csl-9 = Negative X direction
            GPS0 = (53.28989834, -9.07136142, 67.566)  # Origin
            GPSY = (53.28991859, -9.07135091, 67.552)  # Positive Y direction
            GPSX = (53.28991224, -9.07143945, 67.531)  # Negative X direction
            ECEF0 = geoTransformer.gps_to_ecef(*GPS0)
            ECEFX = geoTransformer.gps_to_ecef(*GPSX)
            ECEFY = geoTransformer.gps_to_ecef(*GPSY)
            R = geoTransformer.get_rotation(ECEF0, ECEFX, ECEFY)

        return R

    @staticmethod
    def node_geolocate_object(lidar_origin_gps, compass_heading, object_lidar_coords):
        # Unpack the input tuples
        lat0, lon0 = lidar_origin_gps
        x, y, z = object_lidar_coords
        x = -x

        # Convert compass heading to radians
        heading_rad = math.radians(compass_heading)

        # Compute global offsets
        # Forward (Y in LiDAR) corresponds to North-South direction
        north_offset = y * math.cos(heading_rad) + x * math.sin(heading_rad)
        east_offset = y * math.sin(heading_rad) - x * math.cos(heading_rad)

        # Convert offsets from meters to GPS coordinates
        # Create a geopy point for the origin
        origin = geopy.Point(lat0, lon0)

        # Calculate the new location based on offsets
        north_distance = geopy.distance.distance(meters=north_offset)
        east_distance = geopy.distance.distance(meters=east_offset)

        # Move north and then east
        new_location = north_distance.destination(point=origin, bearing=0)
        new_location = east_distance.destination(point=new_location, bearing=90)

        # Return the GPS coordinates of the detected object
        return new_location.longitude, new_location.latitude, 0

class Plotter:
    @staticmethod
    def format_tick_value(value, pos):
        return f'{value:.2f}'

    @staticmethod
    def generic_plot(ax, x_data, y_data, title, y_label, formatter, delay_node, rotation=45, xlabel=None, color=None, label=None):
        # Plot the data and ensure only one entry is added to the legend per delay_node
        ax.plot(x_data, y_data, label=label, marker='o', color=color)  # Use label if provided

        # Set title and axis labels
        ax.set_title(title)
        ax.set_ylabel(y_label)
        if xlabel:
            ax.set_xlabel(xlabel)

        # Customize axes ticks and format
        ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  # Increase number of x-axis ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # Increase number of y-axis ticks
        ax.xaxis.set_major_formatter(formatter)  # Apply formatting to x-axis
        ax.yaxis.set_major_formatter(formatter)  # Apply formatting to y-axis

        # Enable minor ticks
        plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

        # Rotate x-axis labels if necessary
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)

        # Enable grids
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Display legend with single entry per delay_node
        ax.legend()

    @staticmethod
    def plot_euclidean_distances(eucl_dist_dict):
        small_threshold = 1e-6

        # Set up your plotting subplots here (as you were)
        fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        fig1.subplots_adjust(wspace=0.3)  # Add space between subplots
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        # fig3, ax3 = plt.subplots(figsize=(14, 6))
        # fig4, ax4 = plt.subplots(figsize=(14, 6))
        # fig5, ax5 = plt.subplots(figsize=(14, 6))


        # Formatter for tick values
        formatter = FuncFormatter(Plotter.format_tick_value)

        # Create a color map where each delay_node gets a unique color
        delay_nodes = list(eucl_dist_dict.keys())  # Collect all delay_node values
        color_map = plt.get_cmap('tab10', len(delay_nodes))  # Using a colormap with enough distinct colors
        delay_node_color_map = {delay_node: color_map(i) for i, delay_node in
                                enumerate(delay_nodes)}  # Mapping delay_node to a color

        for delay_node, eucl_dist_df in eucl_dist_dict.items():
            # Filter out rows based on thresholds and non-NaN values
            filtered_eucl_dist_df = eucl_dist_df[
                (np.abs(eucl_dist_df['car_base_to_car_obj']) > small_threshold) &
                (np.abs(eucl_dist_df['node_base_to_node_obj']) > small_threshold) &
                (~eucl_dist_df['car_base_to_car_obj'].isna()) &
                (~eucl_dist_df['node_base_to_node_obj'].isna())]

            # filtered_eucl_dist_df = eucl_dist_df # Uncomment this if you don't want to filter out no detections.

            if filtered_eucl_dist_df.empty:
                print(
                    f"Skipping plotting for delay_node {delay_node}, no data after filtering for no detection/blind-spot from car/node.")
                continue

            time_node = filtered_eucl_dist_df['Time_Node']
            car_base_to_car_obj = filtered_eucl_dist_df['car_base_to_car_obj']
            car_base_to_car_obj_sorted = car_base_to_car_obj.sort_values(ascending=True)

            car_base_to_node_base = filtered_eucl_dist_df['car_base_to_node_base']
            node_base_to_node_obj = filtered_eucl_dist_df['node_base_to_node_obj']
            node_base_to_node_obj_sorted = node_base_to_node_obj.sort_values(ascending=True)

            car_obj_to_node_obj = filtered_eucl_dist_df['car_obj_to_node_obj']
            car_obj_to_node_obj_sorted = car_obj_to_node_obj.loc[car_base_to_car_obj_sorted.index]
            car_obj_to_node_obj_sorted_2 = car_obj_to_node_obj.loc[node_base_to_node_obj_sorted.index]

            # Detect discontinuities based on index differences
            index_diffs = filtered_eucl_dist_df.index.to_series().diff()
            discontinuities = (index_diffs > 1).fillna(False)  # Identify where discontinuties occur
            discont_indexes = np.where(discontinuities)[0]
            segments = np.split(filtered_eucl_dist_df, discont_indexes)

            # Get the color for this delay_node from the color map
            color = delay_node_color_map[delay_node]

            # Track if it's the first plot of this delay_node
            first_plot_of_delay = True

            # Plot each segment separately to avoid connecting lines between discontinuities
            for segment in segments:
                if segment.empty:
                    continue  # Skip empty segments

                # Extract per-segment data
                segment_time_node = segment['Time_Node']
                segment_car_base_to_car_obj = segment['car_base_to_car_obj']
                segment_car_base_to_node_base = segment['car_base_to_node_base']
                segment_node_base_to_node_obj = segment['node_base_to_node_obj']
                segment_car_obj_to_node_obj = segment['car_obj_to_node_obj']

                # Only add a legend label for the first segment of this delay_node
                label = f'Delay Node: {delay_node}ms' if first_plot_of_delay else None

                # After the first plot, set this flag to False to prevent future labels
                first_plot_of_delay = False

                # Plot each segment using the same color for the same delay_node
                Plotter.generic_plot(axs1[0], segment_time_node, segment_car_base_to_car_obj,
                                     title=f'Car Base to Car Object Distance',
                                     y_label='Car Base to Car Object (m)',
                                     formatter=formatter, delay_node=delay_node, rotation=45, color=color, label=label)

                # Set the y-axis and x-axis to start at 0 to remove gaps
                plt.ylim(bottom=0)
                plt.xlim(left=0)

                # Show the plot
                plt.tight_layout()
                plt.show()

                Plotter.generic_plot(axs1[1], segment_time_node, segment_node_base_to_node_obj,
                                     title=f'Node Base to Node Object Distance',
                                     y_label='Node Base to Node Object (m)',
                                     formatter=formatter, delay_node=delay_node, rotation=45, color=color, label=label)

                Plotter.generic_plot(ax2, segment_node_base_to_node_obj, segment_car_obj_to_node_obj,
                                     title=f'Car Object to Node Object Distance',
                                     y_label='Error/Relative Difference (m)',
                                     formatter=formatter, delay_node=delay_node, rotation=45,
                                     xlabel='Distance To Object (m)', color=color, label=label)



                # Plotter.generic_plot(ax3, time_node, car_obj_to_node_obj,
                #                      title=f'Car Object to Node Object Distance',
                #                      y_label='Car Object to Node Object (m)',
                #                      formatter=formatter, delay_node=delay_node, rotation=45,
                #                      xlabel='Node Timestamp', color=color, label=label)
                #
                # Plotter.generic_plot(ax4, car_base_to_car_obj_sorted, car_obj_to_node_obj_sorted,
                #                      title=f'Car Base to Car Object vs Car Object to Node Object',
                #                      y_label='Car Object to Node Object (m)',
                #                      formatter=formatter, delay_node=delay_node, rotation=45,
                #                      xlabel='Car Base to Car Object (m)', color=color, label=label)
                #
                # Plotter.generic_plot(ax5, node_base_to_node_obj, car_obj_to_node_obj_sorted_2,
                #                      title=f'Node Base to Node Object vs Car Object to Node Object',
                #                      y_label='Car Object to Node Object (m)',
                #                      formatter=formatter, delay_node=delay_node, rotation=45,
                #                      xlabel='Node Base to Node Object (m)', color=color, label=label)
        plt.show()

    @staticmethod
    def plot_error_metrics(error_metrics_dict):
        small_threshold = 1e-6

        fig1, ax1 = plt.subplots(figsize=(14, 6))
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        fig3, ax3 = plt.subplots(figsize=(14, 6))

        formatter = FuncFormatter(Plotter.format_tick_value)

        for delay_node, error_metrics_df in error_metrics_dict.items():
            filtered_error_metrics_dist_df = error_metrics_df[np.abs(error_metrics_df['car_base_to_car_obj']) > small_threshold]

            if filtered_error_metrics_dist_df.empty:
                print(f"Skipping plotting for delay_node {delay_node}, no data after filtering for no detection/blind-spot from car/node.")
                continue

            time_node = filtered_error_metrics_dist_df['Time_Node']
            car_obj_to_node_obj = filtered_error_metrics_dist_df['car_obj_to_node_obj']
            metric_1 = filtered_error_metrics_dist_df['metric_1']
            metric_2 = filtered_error_metrics_dist_df['metric_2']

            Plotter.generic_plot(ax1, time_node, car_obj_to_node_obj,
                                 title=f'Car Object to Node Object Distance vs. Node Timestamp',
                                 y_label='Car Object to Node Object (m)',
                                 formatter=formatter, delay_node=delay_node, rotation=45,
                                 xlabel='Node Timestamp')

            Plotter.generic_plot(ax2, time_node, metric_1,
                                 title=f'Error Metric 1 vs. Node Timestamp',
                                 y_label='Error Metric 1',
                                 formatter=formatter, delay_node=delay_node, rotation=45,
                                 xlabel='Node Timestamp')

            Plotter.generic_plot(ax3, time_node, metric_2,
                                 title=f'Error Metric 2 vs. Node Timestamp',
                                 y_label='Error Metric 1',
                                 formatter=formatter, delay_node=delay_node, rotation=45,
                                 xlabel='Node Timestamp')

        plt.show()

    @staticmethod
    def plot_relative_error(relative_error_dict):
        fig, ax = plt.subplots(figsize=(14, 6))
        formatter = FuncFormatter(Plotter.format_tick_value)

        for (delay), rel_error_df in relative_error_dict.items():
            filtered_rel_error_df = rel_error_df[(~rel_error_df['rel_error'].isna())]

            time_car = rel_error_df['Time_Car']
            rel_error = rel_error_df['rel_error']

            ax.plot(time_car, rel_error, label=f'Rel Error: ({delay+100}ms)', marker='o')

        ax.set_title('Relative Error between Car and Node Objects for Consecutive Delays')
        ax.set_xlabel('Car Timestamp')
        ax.set_ylabel('Relative Error (m)')

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

        plt.show()

    @staticmethod
    def plot_delay_diff_interactive(diff_df):
        diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]

        delays = []
        for col in diff_columns:
            try:
                delay = (float(col.split('_')[1].split('-')[1]))  # Convert to float first, then int
                delays.append(delay)
            except ValueError:
                # Ignore columns that cannot be processed
                pass
        fig = go.Figure()

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

        fig.update_layout(
            title='',
            xaxis_title='Heading Error (%)',
            yaxis_title='Difference in Car-to-Node Object Distance (m)',
            legend_title='Rows (Click to toggle)',
            hovermode='x',
            template='plotly',
            xaxis=dict(tickmode='array', tickvals=delays, showgrid=True),
            yaxis=dict(showgrid=True))

        fig.show()