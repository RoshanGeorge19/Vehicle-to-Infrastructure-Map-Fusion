import os
import pandas as pd
import pylas
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
import matplotlib.ticker as ticker
import numpy as np

def read_annotations(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Annotations file not found: {csv_file}")
    return pd.read_csv(csv_file)

def load_las_file(las_path):
    if not os.path.exists(las_path):
        raise FileNotFoundError(f".las file not found: {las_path}")

    las = pylas.read(las_path)
    x = las.X * las.header.scales[0] + las.header.offsets[0]
    y = las.Y * las.header.scales[1] + las.header.offsets[1]
    z = las.Z * las.header.scales[2] + las.header.offsets[2]
    return x, y, z

def is_valid_cuboid(row):
    length, width, height = row['Length'], row['Width'], row['Height']
    x_center, y_center, z_center = row['X_Center'], row['Y_Center'], row['Z_Center']
    return not (length <= 0 or width <= 0 or height <= 0 or
                (x_center == 0 and y_center == 0 and z_center == 0))

def count_points_in_cuboid(x, y, z, row):
    length, width, height = row['Length'], row['Width'], row['Height']
    x_center, y_center, z_center = row['X_Center'], row['Y_Center'], row['Z_Center']

    x_min, x_max = x_center - length / 2, x_center + length / 2
    y_min, y_max = y_center - width / 2, y_center + width / 2
    z_min, z_max = z_center - height / 2, z_center + height / 2

    within_cuboid = (
            (x >= x_min) & (x <= x_max) &
            (y >= y_min) & (y <= y_max) &
            (z >= z_min) & (z <= z_max)
    )
    return within_cuboid.sum()

def calculate_distance(row):
    # Assuming sensor position is at the origin (0, 0, 0); change as needed
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    x_center, y_center, z_center = row['X_Center'], row['Y_Center'], row['Z_Center']
    distance = np.sqrt((x_center - sensor_x)**2 + (y_center - sensor_y)**2 + (z_center - sensor_z)**2)
    return distance

def process_annotations(directory, annotations):
    results = []

    for _, row in annotations.iterrows():
        file_name = row['File_Name']

        if not is_valid_cuboid(row):
            print(f"Skipping invalid cuboid: {file_name}")
            results.append({'File_Name': file_name, 'Num_Points': 0, 'Distance': 0})
            continue

        # Load corresponding .las file
        las_path = os.path.join(directory, file_name)
        try:
            x, y, z = load_las_file(las_path)
        except FileNotFoundError as e:
            print(e)
            results.append({'File_Name': file_name, 'Num_Points': 0, 'Distance': 0})
            continue

        # Count points in cuboid
        num_points = count_points_in_cuboid(x, y, z, row)

        # Calculate distance
        distance = calculate_distance(row)
        results.append({'File_Name': file_name, 'Num_Points': num_points, 'Distance': distance})

    return pd.DataFrame(results)

def format_tick_value(value, pos):
    return f'{value:.2f}'

def plot_points_vs_distance(ax, filtered_df, title):
    formatter = FuncFormatter(format_tick_value)

    ax.plot(filtered_df['Distance'], filtered_df['Num_Points'], color='blue', marker='o', linestyle='-', alpha=0.7)

    ax.set_xlabel('Distance from Sensor to Centroid (m)', fontsize=12)
    ax.set_ylabel('Number of Points', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Customize axes ticks and format
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Set major ticks every 1 meter
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Set minor ticks every 0.1 meters
    ax.xaxis.set_major_formatter(formatter)  # Apply formatting to x-axis

    # Automatically calculate axis range to show exactly 10 ticks
    x_min, x_max = filtered_df['Distance'].min(), filtered_df['Distance'].max()
    ax.set_xlim(np.floor(x_min), np.ceil(x_max))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # Leave up to matplotlib to decide y-ticks
    ax.yaxis.set_major_formatter(formatter)  # Apply formatting to y-axis

    # Rotate x-axis labels by 45 degrees
    ax.tick_params(axis='x', rotation=45)

    ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.7)

def process_and_plot(ax, directory, csv_file, title):
    # Load annotations
    print(f"Reading annotations from: {csv_file}")
    annotations = read_annotations(csv_file)

    # Process annotations to count points
    print(f"Processing LiDAR data from: {directory}")
    results_df = process_annotations(directory, annotations)

    # Merge results with the Distance column
    filtered_df = results_df[results_df['Num_Points'] > 0]
    filtered_df_2 = filtered_df.sort_values(by='Distance')

    # Plot results on the provided axis
    plot_points_vs_distance(ax, filtered_df, title)

def main():
    # Define paths for the first dataset
    directory_1 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/car_cepton_overlap_1/person_1/out/ground_removed'
    csv_file_1 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-755_id-904_fixed_person_1_annotations_car.csv'

    # Define paths for the second dataset
    directory_2 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/car_cepton_overlap_1/person_1/out/ground_removed'
    csv_file_2 = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_fixed_person_1_annotations_node.csv'

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Process and plot for both datasets on subplots
    print("Processing and plotting for the first dataset...")
    # process_and_plot(axs[0], directory_1, csv_file_1, title="Person_1: Car Perspective")
    print("Processing and plotting for the second dataset...")
    process_and_plot(axs[1], directory_2, csv_file_2, title="Person_1: Node Perspective")

    # Display the plots
    plt.show()


if __name__ == "__main__":
    main()