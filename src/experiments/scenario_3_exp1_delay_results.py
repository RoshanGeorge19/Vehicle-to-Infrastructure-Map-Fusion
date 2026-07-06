import numpy as np
import pickle
import matplotlib.pyplot as plt
from wp2.geo_utils import Plotter
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
import os
import pylas
import plotly.graph_objects as go
from collections import defaultdict

def process_annotations(directory, annotations):
    results = []

    for _, row in annotations.iterrows():
        file_name = row['File_Name']
        time = row['Time_Short']
        x_center, y_center, z_center = row['X_Center'], row['Y_Center'], row['Z_Center']
        length, width, height = row['Length'], row['Width'], row['Height']

        if length <= 0 or width <= 0 or height <= 0 or (x_center == 0 and y_center == 0 and z_center == 0):
            # Skip invalid cuboids
            results.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            print(f"Skipped invalid cuboid in file: {file_name}")
            continue

        # Load .las file
        las_path = os.path.join(directory, file_name)
        if not os.path.exists(las_path):
            print(f".las file not found: {las_path}")
            results.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            continue

        las = pylas.read(las_path)
        x = las.X * las.header.scales[0] + las.header.offsets[0]
        y = las.Y * las.header.scales[1] + las.header.offsets[1]
        z = las.Z * las.header.scales[2] + las.header.offsets[2]

        # Count points in cuboid
        x_min, x_max = x_center - length / 2, x_center + length / 2
        y_min, y_max = y_center - width / 2, y_center + width / 2
        z_min, z_max = z_center - height / 2, z_center + height / 2
        num_points = ((x >= x_min) & (x <= x_max) &
                      (y >= y_min) & (y <= y_max) &
                      (z >= z_min) & (z <= z_max)).sum()

        # Calculate distance from sensor
        distance = np.sqrt(x_center ** 2 + y_center ** 2 + z_center ** 2)
        results.append({'File_Name': file_name, 'Time': time, 'Num_Points': num_points, 'Distance': distance})

    return pd.DataFrame(results)

def plot_baseline(eucl_dist_df):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    formatter = FuncFormatter(Plotter.format_tick_value)
    small_threshold = 1e-6

    filtered_eucl_dist_df = eucl_dist_df[(np.abs(eucl_dist_df['car_base_to_car_obj']) > small_threshold) &
                                        (np.abs(eucl_dist_df['node_base_to_node_obj']) > small_threshold) &
                                        (~eucl_dist_df['car_base_to_car_obj'].isna()) &
                                        (~eucl_dist_df['node_base_to_node_obj'].isna())]

    node_base_to_node_obj = filtered_eucl_dist_df['node_base_to_node_obj']
    car_obj_to_node_obj = filtered_eucl_dist_df['car_obj_to_node_obj']
    car_cuboid_points = filtered_eucl_dist_df['Car_Cuboid_Points']
    node_cuboid_points = filtered_eucl_dist_df['Node_Cuboid_Points']

    # columns_to_save = ['car_obj_to_node_obj', 'Car_Cuboid_Points', 'Node_Cuboid_Points', 'node_base_to_node_obj']
    # filtered_data = filtered_eucl_dist_df[columns_to_save]
    # filtered_data.to_csv('saved_data.csv', index=False)

    ax.plot(node_base_to_node_obj, car_obj_to_node_obj, label=f'Node Delay: 0ms', marker='o')
    # ax.set_title('Baseline Error vs Distance', fontsize=16)
    ax.set_ylabel('Euclidean Distance Between Vehicle Detection and Node Detection (m)', fontsize=16)
    ax.set_xlabel('Distance to Object (m)', fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(1)) # Major Tick every 1 meter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.2)) # Minor Tick every 0.2 meter.
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Major Tick every 0.1 meter.
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # Minor Tick every 0.01 meter.
    ax.xaxis.set_major_formatter(formatter)  # Apply 2 decimal places to x-axis
    ax.yaxis.set_major_formatter(formatter)  # Apply 2 decimal places to y-axis
    ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines

    ax1.plot(node_base_to_node_obj, car_cuboid_points, label=f'Node Delay: 0ms', marker='o')
    ax1.set_title('Number of Points in Car Detection vs. Distance to Object', fontsize=16)
    ax1.set_ylabel('Number of Points', fontsize=16)
    ax1.set_xlabel('Distance to Object (m)', fontsize=16)
    ax1.xaxis.set_major_locator(MultipleLocator(1))  # Major Tick every 1 meter.
    ax1.xaxis.set_minor_locator(MultipleLocator(0.2))  # Minor Tick every 0.2 meter.
    ax1.yaxis.set_major_locator(MultipleLocator(5))  # Major Tick every 0.1 meter.
    ax1.yaxis.set_minor_locator(MultipleLocator(1))  # Minor Tick every 0.01 meter.
    ax1.xaxis.set_major_formatter(formatter)  # Apply 2 decimal places to x-axis
    ax1.yaxis.set_major_formatter(formatter)  # Apply 2 decimal places to y-axis
    ax1.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
    ax1.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines

    ax2.plot(node_base_to_node_obj, node_cuboid_points, label=f'Node Delay: 0ms', marker='o')
    ax2.set_title('Number of Points in Node Detection vs. Distance to Object')
    ax2.set_ylabel('Number of Points')
    ax2.set_xlabel('Distance to Object (m)')
    ax2.xaxis.set_major_locator(MultipleLocator(1))  # Major Tick every 1 meter.
    ax2.xaxis.set_minor_locator(MultipleLocator(0.2))  # Minor Tick every 0.2 meter.
    ax2.yaxis.set_major_locator(MultipleLocator(10))  # Major Tick every 0.1 meter.
    ax2.yaxis.set_minor_locator(MultipleLocator(5))  # Minor Tick every 0.01 meter.
    ax2.xaxis.set_major_formatter(formatter)  # Apply 2 decimal places to x-axis
    ax2.yaxis.set_major_formatter(formatter)  # Apply 2 decimal places to y-axis
    ax2.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
    ax2.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    plt.show()

def plot_error_vs_delay(diff_df):
    formatter = FuncFormatter(Plotter.format_tick_value)
    # distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    #                    (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)]
    # range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m', '14-15m',
    #     '15-16m', '16-17m', '17-18m', '18-19m']

    distance_ranges = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18)]
    range_labels = ['8-10m', '10-12m', '12-14m', '14-16m', '16-18m']

    delay_range = np.arange(0, 500, 100)

    # Extract difference columns
    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    delays = []
    for col in diff_columns:
        try:
            delay = float(col.split('_')[1].split('-')[1])  # Extract the delay from the column name
            delays.append(delay)
        except ValueError:
            pass

    # If delay_range is specified, filter the delays and associated columns
    if delay_range is not None:
        filtered_indices = [i for i, d in enumerate(delays) if d in delay_range]
        delays = [delays[i] for i in filtered_indices]
        diff_columns = [diff_columns[i] for i in filtered_indices]

    # Prepare data by range (group rows by distance ranges)
    range_data = {}
    for i, (lower, upper) in enumerate(distance_ranges):
        # Filter rows that fall within the current range
        rows_in_range = diff_df[(diff_df['node_base_to_node_obj'] >= lower) &
                                (diff_df['node_base_to_node_obj'] < upper)]
        if not rows_in_range.empty:
            range_data[range_labels[i]] = rows_in_range

    # Plot individual graphs for each range
    for range_label, data_in_range in range_data.items():
        # Initialize a new figure and axis for each range
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each row in the current range
        for idx, row in data_in_range.iterrows():
            diff_values = [row[col] for col in diff_columns]
            time_car = row['Time_Car'] if 'Time_Car' in row else 'N/A'
            node_base_to_node_obj = row['node_base_to_node_obj']

            ax.plot(delays, diff_values, marker='o',
                    label=f'Row: {idx}, Time: {time_car}, Distance: {node_base_to_node_obj:.2f}m')

        ax.legend(loc='best', fontsize='small')  # Show a legend for trace identification

        ax.set_title('Relative Error vs. Node Delay')
        ax.set_ylabel('Error (m)')
        ax.set_xlabel('Delay (ms)')
        ax.xaxis.set_major_locator(MultipleLocator(100))  # Major Tick every 500ms
        ax.xaxis.set_minor_locator(MultipleLocator(100))  # Minor Tick every 100ms
        ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Major Tick every 1m
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # Minor Tick every 0.1m
        ax.xaxis.set_major_formatter(formatter)  # Apply formatting to x-axis ticks
        ax.yaxis.set_major_formatter(formatter)  # Apply formatting to y-axis ticks
        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines

        fig.tight_layout()
        plt.show()

def plot_error_vs_delay_interactive(diff_df):
    # distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    #                    (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)]

    # distance_ranges = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18)]
    # range_labels = ['8-10m', '10-12m', '12-14m', '14-16m', '16-18m']

    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m']

    delay_range = np.arange(0, 1000, 100)

    # Extract difference columns
    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    delays = []
    for col in diff_columns:
        try:
            delay = float(col.split('_')[1].split('-')[1])  # Extract the delay (e.g., from "Diff_0-100")
            delays.append(delay)
        except ValueError:
            pass

    # If delay_range is specified, filter delays and corresponding columns
    if delay_range is not None:
        filtered_indices = [i for i, d in enumerate(delays) if d in delay_range]
        delays = [delays[i] for i in filtered_indices]
        diff_columns = [diff_columns[i] for i in filtered_indices]

    # Create a Plotly figure
    fig = go.Figure()

    # Add individual row traces
    for idx, row in diff_df.iterrows():
        diff_values = [row[col] for col in diff_columns]
        time_car = row['Time_Car'] if 'Time_Car' in row else 'N/A'
        node_base_to_node_obj = row['node_base_to_node_obj'] if 'node_base_to_node_obj' in row else 'N/A'

        fig.add_trace(go.Scatter(
            x=delays,
            y=diff_values,
            mode='lines+markers',
            name=f'Row: {idx}, Time: {time_car}, Node To Obj Dist: {node_base_to_node_obj}',
            visible='legendonly'  # Initially hidden for clarity
        ))

    # Compute and add range averages
    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])

        # Determine the range label for this row
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(diff_values)

    for range_label, values in range_averages.items():
        avg_values = np.nanmean(values, axis=0)  # Compute the mean across rows
        fig.add_trace(go.Scatter(
            x=delays,
            y=avg_values,
            mode='lines+markers',
            name=f'Average ({range_label})',
            visible=True  # Averages are always visible initially
        ))

    # Update layout
    fig.update_layout(
        title='Difference in Car-to-Node Object Distance (Averaged and Individual Rows)',
        xaxis_title='Delay (ms)',
        yaxis_title='Difference in Car-to-Node Object Distance (m)',
        legend_title='Rows and Averages (Click to toggle)',
        hovermode='x',
        template='plotly',
        xaxis=dict(
            tickmode='array',
            tickvals=delays,
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True
        ),
    )

    # Show the plot
    fig.show()

def plot_error_vs_delay_matplotlib(diff_df):
    distance_ranges = [(8, 9), (9, 10), (10, 11)]
    range_labels = ['8-9m', '9-10m', '10-11m']

    delay_range = np.arange(0, 1000, 100)

    # Extract difference columns
    diff_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    delays = []
    for col in diff_columns:
        try:
            delay = float(col.split('_')[1].split('-')[1])  # Extract the delay (e.g., from "Diff_0-100")
            delays.append(delay)
        except ValueError:
            pass

    # Filter delays and corresponding columns if delay_range is specified
    if delay_range is not None:
        filtered_indices = [i for i, d in enumerate(delays) if d in delay_range]
        delays = [delays[i] for i in filtered_indices]
        diff_columns = [diff_columns[i] for i in filtered_indices]

    # Compute range averages
    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])

        # Determine the range label for this row and group data
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(diff_values)

    plt.figure(figsize=(10, 6))

    # Add range averages to the plot
    for range_label, values in range_averages.items():
        avg_values = np.nanmean(values, axis=0)  # Compute the mean across rows
        plt.plot(delays, avg_values, marker='o', label=f'{range_label}')

    # Configure plot appearance
    plt.title('Impact of Temporal Delay on Spatial Uncertainty', fontsize=16)
    plt.xlabel('Delay (ms)', fontsize=16)
    plt.ylabel('Euclidean Distance Between Vehicle Detection and Node Detection (m)', fontsize=16)
    plt.legend(title='Distance Ranges', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    plotter = Plotter()
    with open('pickles/scenario_3_exp_1.pkl', 'rb') as f:
        data = pickle.load(f)

    directory_1 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/car_cepton_overlap_1/person_1/out/'
    directory_2 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/car_cepton_overlap_1/person_1/out/'
    csv_file_1 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-755_id-904_person_1_annotations_car.csv'
    csv_file_2 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_person_1_annotations_node.csv'
    annotations_1 = pd.read_csv(csv_file_1)
    annotations_2 = pd.read_csv(csv_file_2)
    results_1 = process_annotations(directory_1, annotations_1)
    results_2 = process_annotations(directory_2, annotations_2)

    eucl_dist_dict = data['eucl_dist_dict']
    relative_error_dict = data['relative_error_dict']
    relative_error_diff_df = data['relative_error_diff_df']

    eucl_dist_dict_baseline_car_points = pd.merge(results_1, eucl_dist_dict[0], left_on='Time', right_on='Time_Car')
    eucl_dist_dict_baseline_car_points.rename(columns={'Num_Points': 'Car_Cuboid_Points'}, inplace=True)
    eucl_dist_dict_baseline_car_points.drop(columns=['File_Name', 'Distance', 'Time'], inplace=True)

    eucl_dist_dict_baseline_node_points = pd.merge(results_2, eucl_dist_dict[0], left_on='Time', right_on='Time_Node')
    eucl_dist_dict_baseline_node_points.rename(columns={'Num_Points': 'Node_Cuboid_Points'}, inplace=True)
    eucl_dist_dict_baseline_node_points.drop(columns=['File_Name', 'Distance', 'Time'], inplace=True)

    eucl_dist_dict_baseline = pd.merge(eucl_dist_dict_baseline_car_points[['Car_Cuboid_Points']], eucl_dist_dict_baseline_node_points[['Node_Cuboid_Points']],
        left_index=True, right_index=True)
    eucl_dist_dict_baseline = pd.concat([eucl_dist_dict_baseline, eucl_dist_dict_baseline_car_points.drop(columns=['Car_Cuboid_Points']),
         eucl_dist_dict_baseline_node_points.drop(columns=['Node_Cuboid_Points'])], axis=1)
    eucl_dist_dict_baseline = eucl_dist_dict_baseline.loc[:, ~eucl_dist_dict_baseline.columns.duplicated()]

    plot_baseline(eucl_dist_dict_baseline)
    plot_error_vs_delay_interactive(relative_error_diff_df)
    # plot_error_vs_delay(relative_error_diff_df)
    # plot_error_vs_delay_matplotlib(relative_error_diff_df)

if __name__ == '__main__':
    main()