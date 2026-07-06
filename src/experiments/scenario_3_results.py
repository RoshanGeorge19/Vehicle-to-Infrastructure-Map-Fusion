import os
import numpy as np
import pylas
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from wp2.geo_utils import Plotter, GeoTransformer
from collections import defaultdict
from tqdm import tqdm
from pyproj import Transformer
import rasterio
import laspy
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def rotate_all_coordinates(x_coords, y_coords, z_coords, rotation_matrix):
    # Convert coordinates into a 3×N matrix: [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
    coords = np.vstack((x_coords, y_coords, z_coords))

    # Multiply coordinates by the 3×3 rotation matrix
    rotated_coords = np.dot(rotation_matrix, coords)

    # Extract rotated x, y, and z coordinates
    rotated_x = rotated_coords[0, :]
    rotated_y = rotated_coords[1, :]
    rotated_z = rotated_coords[2, :]

    return rotated_x, rotated_y, rotated_z

def plot_local_map(x_centers_car, y_centers_car, z_centers_car, widths_car, lengths_car,
                   x_centers_node, y_centers_node, z_centers_node, widths_node, lengths_node, max_plot_range):

    def add_radar_guidelines(ax, max_range, num_circles=6):
        """Draw radar-like concentric circles and crosshairs."""
        step_size = max_range / num_circles
        step_size = max(int(np.ceil(step_size)), 1)  # Ensure step size is at least 1

        angles = np.linspace(0, 2 * np.pi, 360)
        radii = np.arange(0, max_range + step_size, step_size)

        for r in radii:
            ax.plot(r * np.cos(angles), r * np.sin(angles), color='gray', lw=0.5, linestyle="--")
            if r != 0:
                ax.text(r, 0.32, f"{int(r)}", color='white', fontsize=10, ha='center', va='center')
                ax.text(-r, 0.32, f"-{int(r)}", color='white', fontsize=10, ha='center', va='center')
                ax.text(0.35, r, f"{int(r)}", color='white', fontsize=10, ha='center', va='center')
                ax.text(0.35, -r, f"-{int(r)}", color='white', fontsize=10, ha='center', va='center')

        ax.plot([-max_range, max_range], [0, 0], color='white', lw=0.7)  # Horizontal crosshair
        ax.plot([0, 0], [-max_range, max_range], color='white', lw=0.7)  # Vertical crosshair
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        step_size = max_range // num_circles or 1
        ticks = np.arange(-max_range, max_range + 1, step_size)

        if 0 in ticks:
            ticks = ticks[ticks != 0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    def add_bounding_boxes(ax, x_centers, y_centers, widths, lengths, color, label):
        """Adds bounding boxes and center points to the plot."""
        for x_center, y_center, width, length in zip(x_centers, y_centers, widths, lengths):
            width = 0.5
            length = 0.25
            rect_x = x_center - width / 2
            rect_y = y_center - length / 2
            rect = plt.Rectangle((rect_x, rect_y), width, length, edgecolor=color, facecolor='none', lw=1)
            ax.add_patch(rect)

        ax.scatter(x_centers, y_centers, color='red', s=8, label="Centers")

        if label:
            ax.text(x_center + 0.5, y_center + 0.2, label, color=color, fontsize=10, ha='left', va='center',
                    weight='bold')

    def create_plot(x_centers, y_centers, widths, lengths, title, max_plot_range):
        """Generates an individual local map plot."""
        max_range = max_plot_range
        num_circles = 6
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')

        add_radar_guidelines(ax, max_range, num_circles)
        add_bounding_boxes(ax, x_centers, y_centers, widths, lengths, color='white', label=None)

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.tight_layout()
        plt.show()

    def create_plot_node_delay(x_centers, y_centers, widths, lengths, title, max_plot_range):
        """Creates a plot that includes both the current and the previous positions."""
        max_range = max_plot_range
        num_circles = 8
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')

        add_radar_guidelines(ax, max_plot_range, num_circles=6)

        # Plots all non-delay bounding boxes
        # add_bounding_boxes(ax, x_centers, y_centers, widths, lengths, color='white')
        legend_entries = []

        # Plots the last bounding box, i.e., nth index.
        if len(x_centers) > 0:
            current_x = x_centers[-1]
            current_y = y_centers[-1]
            current_w = widths[-1] if len(widths) > 0 else 0.5
            current_l = lengths[-1] if len(lengths) > 0 else 0.25
            add_bounding_boxes(ax, [current_x], [current_y], [current_w], [current_l], color='white', label=None)
            legend_entries.append((plt.Rectangle((0, 0), 0.5, 0.25, color='white', lw=4, label=f"Delay: 0ms")))


        delay_colors = ['purple', 'blue', 'cyan', 'green', 'yellow', 'orange']

        delays = np.arange(1, 5, 1) # Delay of 1 time step, i.e., 100ms.

        # Plots all the bounding boxes with delay.
        # previous_x_centers = x_centers[:-delay]
        # previous_y_centers = y_centers[:-delay]

        for delay in delays:
            # Plots the bounding box with delay of the nth index.
            previous_x_centers = [x_centers[-(delay + 1)]] if len(x_centers) > delay else []
            previous_y_centers = [y_centers[-(delay + 1)]] if len(y_centers) > delay else []
            color = delay_colors[(delay - 1) % len(delay_colors)]
            add_bounding_boxes(ax, previous_x_centers, previous_y_centers, widths, lengths, color=color, label=None)

            legend_entries.append((plt.Rectangle((0, 0), 0.5, 0.25, color=color, lw=4, label=f"Delay {delay*100}ms")))

        ax.legend(handles=legend_entries, loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        ax.set_xlim(-max_plot_range, max_plot_range)
        ax.set_ylim(-max_plot_range, max_plot_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.tight_layout()
        plt.show()

    def create_plot_node_xrot(x_centers, y_centers, z_centers, widths, lengths, title, max_plot_range):

        def rotate_point_x(x, y, theta_rad):
            R = np.array([
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad), np.cos(theta_rad)]
            ])
            rotated_point = R @ np.array([x, y])
            return rotated_point[0], rotated_point[1]

        def get_color(index, total_colors):
            cmap = plt.get_cmap("hsv")  # Choose colormap (e.g., "hsv", "rainbow", "tab10", etc.)
            return cmap(index / total_colors)  # Normalize index between 0 and 1

        max_range = max_plot_range
        num_circles = 8
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')

        add_radar_guidelines(ax, max_plot_range, num_circles=6)

        legend_entries = []

        # Retrieve last bounding box, i.e., nth index
        if len(x_centers) > 0:
            current_x = x_centers[-1]
            current_y = y_centers[-1]
            # Plot only the center as a red dot
            ax.scatter([current_x], [current_y], color='white', s=25, label=f"X Rotation: {0}$^\circ$")


        rotation_angles = np.concatenate((np.arange(-10, 0, 2), np.arange(2, 11, 2)))

        total_colors = len(rotation_angles)
        colors = [get_color(i, total_colors) for i in range(total_colors)]

        for idx, rotation_angle in enumerate(rotation_angles):
            theta_rad = np.deg2rad(rotation_angle)  # Convert degrees to radians
            rotated_x, rotated_y = rotate_point_x(current_x, current_y, theta_rad)
            color = colors[idx % len(colors)]
            ax.scatter([rotated_x], [rotated_y], color=color, s=25, label=f"X Rotation: {rotation_angle}$^\circ$")


        ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        ax.set_xlim(-max_plot_range, max_plot_range)
        ax.set_ylim(-max_plot_range, max_plot_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.tight_layout()
        plt.show()

    def create_plot_node_yrot(x_centers, y_centers, z_centers, widths, lengths, title, max_plot_range):

        def rotate_point_y(x, y, z, theta_rad):
            R_y = np.array([
                [np.cos(theta_rad), 0, np.sin(theta_rad)],
                [0, 1, 0],
                [-np.sin(theta_rad), 0, np.cos(theta_rad)]
            ])
            rotated_point = R_y @ np.array([x, y, z])
            return rotated_point[0], rotated_point[1], rotated_point[2]

        def get_color(index, total_colors):
            cmap = plt.get_cmap("hsv")  # Choose colormap (e.g., "hsv", "rainbow", "tab10", etc.)
            return cmap(index / total_colors)  # Normalize index between 0 and 1

        max_range = max_plot_range
        num_circles = 8
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')
        add_radar_guidelines(ax, max_plot_range, num_circles=6)

        # Retrieve last bounding box, i.e., nth index
        if len(x_centers) > 0:
            current_x = x_centers[-1]
            current_y = y_centers[-1]
            current_z = z_centers[-1]
            # Plot only the center as a red dot
            ax.scatter([current_x], [current_y], color='white', s=25, label=f"Y Rotation: {0}$^\circ$")


        rotation_angles = np.concatenate((np.arange(-10, 0, 2), np.arange(2, 11, 2)))

        total_colors = len(rotation_angles)
        colors = [get_color(i, total_colors) for i in range(total_colors)]

        for idx, rotation_angle in enumerate(rotation_angles):
            theta_rad = np.deg2rad(rotation_angle)  # Convert degrees to radians
            rotated_x, rotated_y, rotated_z= rotate_point_y(current_x, current_y, current_z, theta_rad)
            color = colors[idx % len(colors)]
            ax.scatter([rotated_x], [rotated_y], color=color, s=25, label=f"Y Rotation: {rotation_angle}$^\circ$")


        ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        ax.set_xlim(-max_plot_range, max_plot_range)
        ax.set_ylim(-max_plot_range, max_plot_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.tight_layout()
        plt.show()

    def create_plot_node_zrot(x_centers, y_centers, z_centers, widths, lengths, title, max_plot_range):
        def rotate_point_y(x, y, z, theta_rad):
            R_z = np.array([
                [np.cos(theta_rad), -np.sin(theta_rad), 0],
                [np.sin(theta_rad), np.cos(theta_rad), 0],
                [0, 0, 1]
            ])

            # Apply the rotation
            rotated_point = R_z @ np.array([x, y, z])
            return rotated_point[0], rotated_point[1], rotated_point[2]

        def get_color(index, total_colors):
            cmap = plt.get_cmap("hsv")  # Choose colormap (e.g., "hsv", "rainbow", "tab10", etc.)
            return cmap(index / total_colors)  # Normalize index between 0 and 1

        max_range = max_plot_range
        num_circles = 8
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')
        add_radar_guidelines(ax, max_plot_range, num_circles=6)

        # Retrieve last bounding box, i.e., nth index
        if len(x_centers) > 0:
            current_x = x_centers[-1]
            current_y = y_centers[-1]
            current_z = z_centers[-1]
            # Plot only the center as a red dot
            ax.scatter([current_x], [current_y], color='white', s=25, label=f"Z Rotation: {0}$^\circ$")


        rotation_angles = np.concatenate((np.arange(-10, 0, 2), np.arange(2, 11, 2)))

        total_colors = len(rotation_angles)
        colors = [get_color(i, total_colors) for i in range(total_colors)]

        for idx, rotation_angle in enumerate(rotation_angles):
            theta_rad = np.deg2rad(rotation_angle)  # Convert degrees to radians
            rotated_x, rotated_y, rotated_z= rotate_point_y(current_x, current_y, current_z, theta_rad)
            color = colors[idx % len(colors)]
            ax.scatter([rotated_x], [rotated_y], color=color, s=25, label=f"Z Rotation: {rotation_angle}$^\circ$")


        ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        ax.set_xlim(-max_plot_range, max_plot_range)
        ax.set_ylim(-max_plot_range, max_plot_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.tight_layout()
        plt.show()

    def create_plot_node_xtran(x_centers, y_centers, z_centers, widths, lengths, title, max_plot_range):
        def translate_point_x(x, y, z, translation):
            translated_point = np.array([translation, 0, 0]) + np.array([x, y, z])
            return translated_point[0], translated_point[1], translated_point[2]

        def get_color(index, total_colors):
            cmap = plt.get_cmap("hsv")  # Choose colormap (e.g., "hsv", "rainbow", "tab10", etc.)
            return cmap(index / total_colors)  # Normalize index between 0 and 1

        max_range = max_plot_range
        num_circles = 8
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')
        add_radar_guidelines(ax, max_plot_range, num_circles=6)

        # Retrieve last bounding box, i.e., nth index
        if len(x_centers) > 0:
            current_x = x_centers[-1]
            current_y = y_centers[-1]
            current_z = z_centers[-1]
            # Plot only the center as a red dot
            ax.scatter([current_x], [current_y], color='white', s=25, label=f"X Translation: {0}m")


        trans_values = np.concatenate((np.arange(-5, 0, 1), np.arange(1, 5, 1)))

        total_colors = len(trans_values)
        colors = [get_color(i, total_colors) for i in range(total_colors)]

        for idx, translation_val in enumerate(trans_values):
            trans_x, trans_y, trans_z = translate_point_x(current_x, current_y, current_z, translation_val)
            color = colors[idx % len(colors)]
            ax.scatter([trans_x], [trans_y], color=color, s=25, label=f"X Translation: {translation_val}m")


        ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        ax.set_xlim(-max_plot_range, max_plot_range)
        ax.set_ylim(-max_plot_range, max_plot_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.tight_layout()
        plt.show()

    def create_plot_node_ytran(x_centers, y_centers, z_centers, widths, lengths, title, max_plot_range):
        def translate_point_y(x, y, z, translation):
            translated_point = np.array([0, translation, 0]) + np.array([x, y, z])
            return translated_point[0], translated_point[1], translated_point[2]

        def get_color(index, total_colors):
            cmap = plt.get_cmap("hsv")  # Choose colormap (e.g., "hsv", "rainbow", "tab10", etc.)
            return cmap(index / total_colors)  # Normalize index between 0 and 1

        max_range = max_plot_range
        num_circles = 8
        step_size = max(int(np.ceil(max_range / num_circles)), 1)
        ticks = np.arange(-max_range, max_range + step_size, step_size)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')
        add_radar_guidelines(ax, max_plot_range, num_circles=6)

        # Retrieve last bounding box, i.e., nth index
        if len(x_centers) > 0:
            current_x = x_centers[-1]
            current_y = y_centers[-1]
            current_z = z_centers[-1]
            # Plot only the center as a red dot
            ax.scatter([current_x], [current_y], color='white', s=25, label=f"Y Translation: {0}m")


        trans_values = np.concatenate((np.arange(-5, 0, 1), np.arange(1, 5, 1)))

        total_colors = len(trans_values)
        colors = [get_color(i, total_colors) for i in range(total_colors)]

        for idx, translation_val in enumerate(trans_values):
            trans_x, trans_y, trans_z = translate_point_y(current_x, current_y, current_z, translation_val)
            color = colors[idx % len(colors)]
            ax.scatter([trans_x], [trans_y], color=color, s=25, label=f"Y Translation: {translation_val}m")


        ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        ax.set_xlim(-max_plot_range, max_plot_range)
        ax.set_ylim(-max_plot_range, max_plot_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.tight_layout()
        plt.show()


    # ---------- Call the function twice for two separate plots ----------
    # create_plot(x_centers_car, y_centers_car, widths_car, lengths_car, "Vehicle Local Map", max_plot_range)
    # create_plot(x_centers_node, y_centers_node, widths_node, lengths_node, "Node Local Map", max_plot_range)
    # create_plot_node_delay(x_centers_node, y_centers_node, widths_node, lengths_node, "Node Local Map with Delay", max_plot_range)
    # create_plot_node_xrot(x_centers_node, y_centers_node, z_centers_node, widths_node, lengths_node, "Node Local Map with X Rotation", max_plot_range)
    # create_plot_node_yrot(x_centers_node, y_centers_node, z_centers_node, widths_node, lengths_node, "Node Local Map with Y Rotation", max_plot_range)
    # create_plot_node_zrot(x_centers_node, y_centers_node, z_centers_node, widths_node, lengths_node, "Node Local Map with Z Rotation", max_plot_range)
    # create_plot_node_xtran(x_centers_node, y_centers_node, z_centers_node, widths_node, lengths_node, "Node Local Map with X Translation", max_plot_range)
    # create_plot_node_ytran(x_centers_node, y_centers_node, z_centers_node, widths_node, lengths_node, "Node Local Map with Y Translation", max_plot_range=15)

def plot_csv_data(csv_file_path: str, range_name: str = '', label: str = ''):
    """
    Plots data from a CSV file with 'index' on the x-axis and 'difference' on the y-axis,
    applying the specified matplotlib styling.

    Args:
        csv_file_path (str): The path to the CSV file.
        range_name (str): The name of the range to be displayed in the title.
        label (str): The label for the plotted data, used in the legend.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'Index' not in df.columns or 'Difference' not in df.columns:
        print("Error: CSV file must contain 'index' and 'difference' columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Index'], df['Difference']*1000, label=label)

    # # Apply styling from the snippet
    # ax.set_title(f'Impact of X-Axis Translation Error on Spatial Uncertainty: {range_name}', fontsize=16)
    ax.set_xlabel('LiDAR Data Index', fontsize=18)
    ax.set_ylabel('Time Difference (ms)', fontsize=18)
    #
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # formatter = ticker.FormatStrFormatter('%.2f')
    # ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=18)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha="right", fontsize=18)
    plt.setp(ax.yaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
    # ax.yaxis.set_major_formatter(formatter)
    #
    ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    # ax.legend(title='Averaged Distance Ranges', fontsize=14)

    plt.savefig(
        'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
        'scenario_3_baseline/sc3_node_car_time_diff.png',
        bbox_inches='tight', dpi=400, format='png')

    plt.tight_layout()
    plt.show()


def plot_global_map(geotiff_path, global_baseline_csv):
    df = pd.read_csv(global_baseline_csv)
    df_filtered = df[df['Show'] == 1]

    with rasterio.open(geotiff_path) as dataset:
        if dataset.count < 3:
            raise ValueError("GeoTIFF doesn't have enough bands for RGB display.")

        # Read the Red, Green, and Blue bands (typically bands 1, 2, 3)
        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)

        # Stack the bands into an (R, G, B) image
        rgb_image = np.dstack((red, green, blue))

        # Get the extent of the GeoTIFF image in the dataset's CRS (left, right, bottom, top)
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Get the affine transform and CRS
        transform = dataset.transform
        crs = dataset.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image, extent=extent)

        node_colors = {}
        node_color_list = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'pink', 'gray', 'olive', 'navy']

        # Get unique x values from NODE_x_Ti
        unique_x_values = sorted(
            set(
                int(parts[1]) for name in df_filtered['Name']
                if 'node' in name.lower() and (parts := name.split('_'))[1].isdigit()
            )
        )

        for i, x in enumerate(unique_x_values):
            node_colors[x] = node_color_list[i % len(node_color_list)]

        legend_handles = []
        legend_labels = set()  # Prevents duplicate legend entries

        for _, row in df_filtered.iterrows():
            lon, lat, name = row['Longitude'], row['Latitude'], row['Name']

            center_lon = lon
            center_lat = lat

            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))

            rect_width = 0.5 / meters_per_degree_lon
            rect_height = 0.25 / meters_per_degree_lat

            rect_x = center_lon - rect_width / 2
            rect_y = center_lat - rect_height / 2

            if 'car_base' in name.lower():
                point_color = 'red'
                label_name = 'Vehicle Base'
                marker = 's'

                if label_name not in legend_labels:
                    legend_labels.add(label_name)
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=point_color, markersize=7, linestyle='', label=label_name))

                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            elif 'node_base' in name.lower():
                point_color = 'black'
                label_name = 'Node Base'
                marker = 's'

                if label_name not in legend_labels:
                    legend_labels.add(label_name)
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=point_color, markersize=7, linestyle='', label=label_name))

                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            elif 'car' in name.lower():
                point_color = 'red'
                marker = 'o'
                label_name = 'Vehicle Detection'

                if label_name not in legend_labels:
                    legend_labels.add(label_name)
                    legend_handles.append(
                        Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='', label=label_name))

            elif 'node' in name.lower():
                try:
                    x_value = int(name.split('_')[1])
                    # point_color = node_colors.get(x_value, 'black')
                    point_color = 'blue'
                    label_name = f'Node Detection'
                    marker = 'o'

                    if label_name not in legend_labels:
                        legend_labels.add(label_name)
                        legend_handles.append(
                            Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='',
                                   label=label_name))

                except (IndexError, ValueError):
                    point_color = 'black'
                    label_name = 'Node'

                    if label_name not in legend_labels:
                        legend_labels.add(label_name)
                        legend_handles.append(
                            Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='',
                                   label=label_name))

            if 'car_base' not in name.lower() and 'node_base' not in name.lower():
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=3.5, color=point_color)

        ax.set_xlabel('Map X Coordinate (CRS)')
        ax.set_ylabel('Map Y Coordinate (CRS)')
        ax.set_title('GeoTIFF RGB Image with Labeled Data Points')

        ax.legend(handles=legend_handles, loc='upper left', title="Legend")

        plt.grid(True)
        plt.show()

    return None

def plot_delay_global_map_old(geotiff_path, global_baseline_csv):
    df = pd.read_csv(global_baseline_csv)
    df_filtered = df[df['Show'] == 1]

    with rasterio.open(geotiff_path) as dataset:
        if dataset.count < 3:
            raise ValueError("GeoTIFF doesn't have enough bands for RGB display.")

        # Read the Red, Green, and Blue bands (typically bands 1, 2, 3)
        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)

        # Stack the bands into an (R, G, B) image
        rgb_image = np.dstack((red, green, blue))

        # Get the extent of the GeoTIFF image in the dataset's CRS
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Get the transform and CRS
        transform = dataset.transform
        crs = dataset.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image, extent=extent)

        node_colors = {}

        # Define a Fixed List of Colors
        node_color_list = ['blue', 'green', 'purple', 'orange', 'brown', 'yellow', 'pink', 'gray', 'olive', 'navy']

        # Get unique x values from NODE_x_Ti
        unique_x_values = sorted(
            set(
                int(parts[1]) for name in df_filtered['Name']
                if 'node' in name.lower() and (parts := name.split('_'))[1].isdigit()
            )
        )

        # Assign colors based on index (cycle through the list if needed)
        for i, x in enumerate(unique_x_values):
            if x == 0:
                node_colors[x] = 'black'  # Set color to black if x is 0
            else:
                node_colors[x] = node_color_list[i % len(node_color_list)]

        legend_handles = []  # To store legend handles

        for _, row in df_filtered.iterrows():
            lon, lat, name = row['Longitude'], row['Latitude'], row['Name']

            # Compute the center of the image
            center_lon = lon
            center_lat = lat

            meters_per_degree_lat = 111320  # Approximate meters per degree
            meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))  # Adjust for latitude

            rect_width = 0.5 / meters_per_degree_lon  # Convert 0.5m to degrees longitude
            rect_height = 0.25 / meters_per_degree_lat  # Convert 0.25m to degrees latitude

            rect_x = center_lon - rect_width / 2
            rect_y = center_lat - rect_height / 2

            if 'car_base' in name.lower():
                point_color = 'red'
                label_name = 'Vehicle Location'
                marker = 's'

                if not any(handle.get_label() == 'Vehicle Location' for handle in legend_handles):
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=point_color, markersize=7, linestyle='', label=label_name))
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            elif 'node_base' in name.lower():
                point_color = 'black'
                label_name = 'Node Location'
                marker = 's'

                if not any(handle.get_label() == 'Node Location' for handle in legend_handles):
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=point_color, markersize=7, linestyle='', label=label_name))

                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)


            elif 'car' in name.lower():
                point_color = 'red'
                marker = 'o'  # Circle for movable vehicles

                if not any(handle.get_label() == 'Car' for handle in legend_handles):
                    legend_handles.append(
                        Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='', label='Vehicle Detection')
                    )


            elif 'node' in name.lower():
                try:
                    x_value = int(name.split('_')[1])  # Extract delay time (x value) for Node
                    point_color = node_colors.get(x_value, 'black')  # Default to black
                    label_name = f'Node Delay: {x_value}ms'
                    marker = 'o'

                    if not any(handle.get_label() == label_name for handle in legend_handles):
                        legend_handles.append(
                            Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='', label=label_name)
                        )

                except (IndexError, ValueError):
                    point_color = 'black'  # Default color if parsing fails
                    label_name = 'Node'

            # Plot all other points (cars, nodes) except car_base and node_base since they were already plotted above
            if 'car_base' not in name.lower() and 'node_base' not in name.lower():
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=3.5, color=point_color)

        # 📌 Final Labels and Legend
        ax.set_xlabel('Map X Coordinate (CRS)')
        ax.set_ylabel('Map Y Coordinate (CRS)')
        ax.set_title('GeoTIFF RGB Image with Vehicle and Node Locations')

        # 🏁 Ensure legend correctly represents all elements
        ax.legend(handles=legend_handles, loc='upper left', title="Legend")

        plt.grid(True)
        plt.show()

    return None

def plot_delay_global_map(geotiff_path, global_baseline_csv):
    # Read data
    df = pd.read_csv(global_baseline_csv)
    df_filtered = df[df['Show'] == 1]

    with rasterio.open(geotiff_path) as dataset:
        if dataset.count < 3:
            raise ValueError("GeoTIFF doesn't have enough bands for RGB display.")
        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)
        rgb_image = np.dstack((red, green, blue))
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        transform = dataset.transform
        crs = dataset.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image, extent=extent)

        # Viridis colormap setup
        viridis = plt.get_cmap('copper')
        norm = mcolors.Normalize(vmin=0, vmax=900)
        legend_handles = []

        for _, row in df_filtered.iterrows():
            lon, lat, name = row['Longitude'], row['Latitude'], row['Name']
            center_lon = lon
            center_lat = lat

            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
            rect_width = 0.5 / meters_per_degree_lon
            rect_height = 0.25 / meters_per_degree_lat
            rect_x = center_lon - rect_width / 2
            rect_y = center_lat - rect_height / 2

            map_x, map_y = transformer.transform(lon, lat)

            # CASE 1: Vehicle Static Location
            if 'car_base' in name.lower():
                point_color = 'red'
                label_name = 'Vehicle Location'
                marker = 's'
                if not any(h.get_label() == label_name for h in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker='s', color=point_color, markersize=7,
                                                 linestyle='', label=label_name))
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            # CASE 2: Node Static Location
            elif 'node_base' in name.lower():
                point_color = 'black'
                label_name = 'Node Location'
                marker = 's'
                if not any(h.get_label() == label_name for h in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker='s', color=point_color, markersize=7,
                                                 linestyle='', label=label_name))
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            # CASE 3:
            elif 'car' in name.lower():
                point_color = 'red'
                label_name = 'Vehicle Detection'
                marker = 'o'
                if not any(h.get_label() == label_name for h in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker='o', color=point_color, markersize=7,
                                                 linestyle='', label=label_name))

            # CASE 4:
            elif 'node' in name.lower():
                try:
                    x_value = int(name.split('_')[1])  # Extract delay (ms)
                    rgba_color = viridis(norm(x_value))
                    point_color = rgba_color
                    label_name = f'Node Delay: {x_value}ms'
                    marker = 'o'
                    if not any(h.get_label() == label_name for h in legend_handles):
                        legend_handles.append(Line2D([0], [0], marker='o', color=rgba_color, markersize=7,
                                                     linestyle='', label=label_name))
                except (IndexError, ValueError):
                    point_color = 'black'
                    label_name = 'Node'

            # Plot the point if not static car/node
            if 'car_base' not in name.lower() and 'node_base' not in name.lower():
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

        # Labels and legend
        ax.set_xlabel('Map X Coordinate (CRS)')
        ax.set_ylabel('Map Y Coordinate (CRS)')
        ax.set_title('GeoTIFF RGB Image with Vehicle and Node Locations')

        ax.legend(handles=legend_handles, loc='upper left', title="Legend")
        plt.grid(True)
        plt.show()

    return None

def plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title):
    df = pd.read_csv(global_baseline_csv)
    df_filtered = df[df['Show'] == 1]

    with rasterio.open(geotiff_path) as dataset:
        if dataset.count < 3:
            raise ValueError("GeoTIFF doesn't have enough bands for RGB display.")

        # Read RGB bands
        red, green, blue = dataset.read(1), dataset.read(2), dataset.read(3)
        rgb_image = np.dstack((red, green, blue))

        # Get image extent
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Set up coordinate transformation
        crs = dataset.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image, extent=extent)

        # Define a fixed list of colors
        node_color_list = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'pink', 'gray', 'olive', 'navy']
        node_color_list = ['blue', 'purple', 'green', 'purple', 'brown', 'cyan', 'pink', 'gray', 'olive', 'navy']
        node_colors = {}

        # Extract unique x values from NODE_x_Ti names
        unique_x_values = sorted(
            set(float(parts[1]) for name in df_filtered['Name']
                if 'node' in name.lower() and (parts := name.split('_'))[1].replace('.', '', 1).lstrip('-').isdigit()))

        # Assign colors based on index (cycle through the list if needed)
        for i, x in enumerate(unique_x_values):
            if x == 0.0:
                node_colors[x] = 'black'  # Set color to black if x is exactly 0
            else:
                node_colors[x] = node_color_list[i % len(node_color_list)]

        legend_handles = []  # To store legend handles

        for _, row in df_filtered.iterrows():
            lon, lat, name = row['Longitude'], row['Latitude'], row['Name']

            # Compute the center of the image
            center_lon, center_lat = lon, lat
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
            rect_width, rect_height = 0.5 / meters_per_degree_lon, 0.25 / meters_per_degree_lat
            rect_x, rect_y = center_lon - rect_width / 2, center_lat - rect_height / 2

            if 'car_base' in name.lower():
                pass
                point_color, marker, label_name = 'red', 's', 'Vehicle Location'
                if not any(handle.get_label() == 'Vehicle Base' for handle in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker='s', color=point_color, markersize=7, linestyle='', label=label_name))
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            elif 'node_base' in name.lower():
                pass
                point_color, marker, label_name = 'black', 's', 'Node Location'
                if not any(handle.get_label() == 'Node Base' for handle in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker='s', color=point_color, markersize=7, linestyle='', label=label_name))
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            elif 'car' in name.lower():
                point_color, marker = 'red', 'o'
                if not any(handle.get_label() == 'Vehicle Detection' for handle in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='', label='Vehicle Detection'))

            elif 'node' in name.lower():
                try:
                    x_value = float(name.split('_')[1])  # Extract x value as float
                    point_color = node_colors.get(x_value, 'black')  # Default to black
                    label_name = f'{label_title}: {x_value}$^\circ$'
                    label_name = f'{label_title}: {x_value}m'
                    marker = 'o'

                    if not any(handle.get_label() == label_name for handle in legend_handles):
                        legend_handles.append(Line2D([0], [0], marker='o', color=point_color, markersize=7, linestyle='', label=label_name))

                except (IndexError, ValueError):
                    point_color = 'black'  # Default color if parsing fails
                    label_name = 'Node'

            # Plot all other points except already handled car_base and node_base
            if 'car_base' not in name.lower() and 'node_base' not in name.lower():
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=6.5, color=point_color) # change markersize to 6.5 for zoomed in roi.

        # Labels and Legend
        ax.set_xlabel('Map X Coordinate (CRS)')
        ax.set_ylabel('Map Y Coordinate (CRS)')
        ax.set_title('GeoTIFF RGB Image with Vehicle and Node Locations')

        ax.legend(handles=legend_handles, loc='upper left', title="Legend")

        plt.grid(True)
        plt.show()

    return None

def plot_heading_rotation_global_map_2(geotiff_path, global_baseline_csv, label_title):
    use_viridis_colormap = True  # Set to False to use predefined colors

    df = pd.read_csv(global_baseline_csv)
    df_filtered = df[df['Show'] == 1]

    with rasterio.open(geotiff_path) as dataset:
        if dataset.count < 3:
            raise ValueError("GeoTIFF doesn't have enough bands for RGB display.")

        red, green, blue = dataset.read(1), dataset.read(2), dataset.read(3)
        rgb_image = np.dstack((red, green, blue))

        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        crs = dataset.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image, extent=extent)

        # 🎨 Predefined colors
        node_color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'pink', 'gray', 'olive', 'navy']
        node_colors = {}

        # 🎯 Get the unique "x values" from node_x_Ti patterns
        unique_x_values = sorted(
            set(float(parts[1]) for name in df_filtered['Name']
                if 'node' in name.lower() and (parts := name.split('_'))[1].replace('.', '', 1).lstrip('-').isdigit())
        )

        # 🟢 Colormap setup (viridis)
        viridis = plt.get_cmap('inferno')
        norm = mcolors.Normalize(vmin=min(unique_x_values), vmax=max(unique_x_values))

        # 🎨 Assign color based on mode
        for i, x in enumerate(unique_x_values):
            if use_viridis_colormap:
                node_colors[x] = viridis(norm(x))  # RGBA from colormap
            else:
                if x == 0.0:
                    node_colors[x] = 'black'
                else:
                    node_colors[x] = node_color_list[i % len(node_color_list)]

        legend_handles = []

        for _, row in df_filtered.iterrows():
            lon, lat, name = row['Longitude'], row['Latitude'], row['Name']
            center_lon, center_lat = lon, lat

            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
            rect_width, rect_height = 0.5 / meters_per_degree_lon, 0.25 / meters_per_degree_lat
            rect_x, rect_y = center_lon - rect_width / 2, center_lat - rect_height / 2

            # 🚗 Vehicle base
            if 'car_base' in name.lower():
                point_color, marker, label_name = 'red', 's', 'Vehicle Location'
                if not any(handle.get_label() == label_name for handle in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker=marker, color=point_color, markersize=7, linestyle='', label=label_name))
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            # 📡 Node base
            elif 'node_base' in name.lower():
                point_color, marker, label_name = 'black', 's', 'Node Location'
                if not any(handle.get_label() == label_name for handle in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker=marker, color=point_color, markersize=7, linestyle='', label=label_name))
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=5, color=point_color)

            # 🚘 Moving car
            elif 'car' in name.lower():
                point_color, marker = 'red', 'o'
                if not any(handle.get_label() == 'Vehicle Detection' for handle in legend_handles):
                    legend_handles.append(Line2D([0], [0], marker=marker, color=point_color, markersize=7, linestyle='', label='Vehicle Detection'))

            # 🟣 Node Condition Point
            elif 'node' in name.lower():
                try:
                    x_value = float(name.split('_')[1])  # Extract float value
                    point_color = node_colors.get(x_value, 'black')
                    label_name = f'{label_title}: {x_value}m'
                    marker = 'o'

                    if not any(h.get_label() == label_name for h in legend_handles):
                        legend_handles.append(Line2D([0], [0], marker=marker, color=point_color, markersize=7, linestyle='', label=label_name))

                except (IndexError, ValueError):
                    point_color = 'black'
                    label_name = 'Node'

            if 'car_base' not in name.lower() and 'node_base' not in name.lower():
                map_x, map_y = transformer.transform(lon, lat)
                ax.plot(map_x, map_y, marker=marker, markersize=6.5, color=point_color)

        # 📌 Final details
        ax.set_xlabel('Map X Coordinate (CRS)')
        ax.set_ylabel('Map Y Coordinate (CRS)')
        ax.set_title('GeoTIFF RGB Image with Vehicle and Node Locations')
        ax.legend(handles=legend_handles, loc='upper left', title="Legend")
        plt.grid(True)
        plt.show()

    return None

def plot_point_global(geo_transformer, geotiff_path, output_csv_path, node_gps, node_annotations,
                                       node_point_path, compass_heading, df_R_t_interval, car_gps, vehicle_annotations, car_point_path):

    node_point_clouds = [file for file in os.listdir(node_point_path) if os.path.isfile(os.path.join(node_point_path, file))]
    car_point_clouds = [file for file in os.listdir(car_point_path) if os.path.isfile(os.path.join(car_point_path, file))]

    data_node = pd.read_csv(node_annotations)
    data_car = pd.read_csv(vehicle_annotations)

    for node_file in node_point_clouds:
        for car_file in car_point_clouds:
            node_las_file_path = os.path.join(node_point_path, node_file)
            node_inFile = laspy.read(node_las_file_path)
            node_points = (node_inFile.x, node_inFile.y, node_inFile.z)
            n_x, n_y, n_z = node_points

            car_las_file_path = os.path.join(car_point_path, car_file)
            car_inFile = laspy.read(car_las_file_path)
            car_points = (car_inFile.x, car_inFile.y, car_inFile.z)
            c_x, c_y, c_z = car_points

            node_data = data_node[data_node['File_Name'] == node_file]
            n_x_center = node_data['X_Center'].values[0]
            n_y_center = node_data['Y_Center'].values[0]
            n_z_center = node_data['Z_Center'].values[0]
            n_length = node_data['Length'].values[0]
            n_width = node_data['Width'].values[0]
            n_height = node_data['Height'].values[0]

            car_data = data_car[data_car['File_Name'] == car_file]
            c_x_center = car_data['X_Center'].values[0]
            c_y_center = car_data['Y_Center'].values[0]
            c_z_center = car_data['Z_Center'].values[0]
            c_length = car_data['Length'].values[0]
            c_width = car_data['Width'].values[0]
            c_height = car_data['Height'].values[0]

            # Compute cuboid boundaries
            n_x_min, n_x_max = n_x_center - n_length / 2, n_x_center + n_length / 2
            n_y_min, n_y_max = n_y_center - n_width / 2, n_y_center + n_width / 2
            n_z_min, n_z_max = n_z_center - n_height / 2, n_z_center + n_height / 2

            # Compute cuboid boundaries
            c_x_min, c_x_max = c_x_center - c_length / 2, c_x_center + c_length / 2
            c_y_min, c_y_max = c_y_center - c_width / 2, c_y_center + c_width / 2
            c_z_min, c_z_max = c_z_center - c_height / 2, c_z_center + c_height / 2

            # Filter points within the cuboid
            n_mask = (n_x >= n_x_min) & (n_x <= n_x_max) & \
                   (n_y >= n_y_min) & (n_y <= n_y_max) & \
                   (n_z >= n_z_min) & (n_z <= n_z_max)

            # Filter points within the cuboid
            c_mask = (c_x >= c_x_min) & (c_x <= c_x_max) & \
                     (c_y >= c_y_min) & (c_y <= c_y_max) & \
                     (c_z >= c_z_min) & (c_z <= c_z_max)

            n_x_filtered = n_x[n_mask]
            n_y_filtered = n_y[n_mask]
            n_z_filtered = n_z[n_mask]

            c_x_filtered = c_x[c_mask]
            c_y_filtered = c_y[c_mask]
            c_z_filtered = c_z[c_mask]

            gps_data_list = []

            for i, (n_x_f, n_y_f, n_z_f) in enumerate(zip(n_x_filtered, n_y_filtered, n_z_filtered)):
                n_lidar_point = np.array([n_x_f, n_y_f, n_z_f])
                n_gps_point = geo_transformer.node_geolocate_object(node_gps, compass_heading, n_lidar_point)
                gps_data_list.append([100,  f"pt_node_{i}", n_gps_point[0],  n_gps_point[1],  0,  1])

            for i, (c_x_f, c_y_f, c_z_f) in enumerate(zip(c_x_filtered, c_y_filtered, c_z_filtered)):
                c_lidar_point = np.array([c_x_f, c_y_f, c_z_f])

                theta_lidar_to_local = np.deg2rad(-15)
                R_LidarToLocal = [[np.cos(theta_lidar_to_local), -np.sin(theta_lidar_to_local), 0],
                                  [np.sin(theta_lidar_to_local), np.cos(theta_lidar_to_local), 0],
                                  [0, 0, 1]]

                R_LocalToGlobal = geo_transformer.getLidarToLocalCS_Rotation(df_R_t_interval, 0, geo_transformer, hard_values=True)

                point_local_cs = np.dot(R_LidarToLocal, c_lidar_point)
                point_local_cs_left_neg = np.array([-point_local_cs[0], point_local_cs[1], point_local_cs[2]])

                ECEF_base_curr = geo_transformer.gps_to_ecef(*car_gps)
                point_global_cs_ecef = geo_transformer.lidar_to_ecef(point_local_cs_left_neg, ECEF_base_curr, R_LocalToGlobal)
                c_gps_point = geo_transformer.ecef_to_gps(*point_global_cs_ecef)
                gps_data_list.append([100,  f"pt_car_{i}", c_gps_point[0],  c_gps_point[1],  0,  1])

            gps_df = pd.DataFrame(gps_data_list, columns=["Colour_ID", "Name", "Longitude", "Latitude", "Altitude", "Show"])

            open(output_csv_path, 'w').close()
            gps_df.to_csv(output_csv_path, index=False)
            # To Do.
            # plot_individual_point_global_map(geotiff_path, output_csv_path)

    return None

def node_annotation_quality():
    node_data_directory = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/car_cepton_overlap_1/person_1/out/'
    node_annotation_csv = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_fixed_person_1_annotations_node.csv'

    node_annotations = pd.read_csv(node_annotation_csv)
    results_original = []
    results_shifted = []

    for _, row in node_annotations.iterrows():
        file_name = row['File_Name']
        time = row['Time_Short']

        # ORIGINAL CUBOID
        length, width, height = row['Length'], row['Width'], row['Height']
        x_center, y_center, z_center = row['X_Center'], row['Y_Center'], row['Z_Center']

        # SHIFTED CUBOID (Fixed 1x1x1m)
        x_center_shifted, z_center_shifted = x_center - 1, z_center - 1.5
        length_shifted, width_shifted, height_shifted = 1, 1, 1

        if length <= 0 or width <= 0 or height <= 0 or (x_center == 0 and y_center == 0 and z_center == 0):
            results_original.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            results_shifted.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            print(f"Skipped invalid cuboid in file: {file_name}")
            continue

        las_path = os.path.join(node_data_directory, file_name)
        if not os.path.exists(las_path):
            print(f".las file not found: {las_path}")
            results_original.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            results_shifted.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            continue

        las = pylas.read(las_path)
        x = las.X * las.header.scales[0] + las.header.offsets[0]
        y = las.Y * las.header.scales[1] + las.header.offsets[1]
        z = las.Z * las.header.scales[2] + las.header.offsets[2]

        # Original Cuboid
        x_min, x_max = x_center - length / 2, x_center + length / 2
        y_min, y_max = y_center - width / 2, y_center + width / 2
        z_min, z_max = z_center - height / 2, z_center + height / 2
        num_points_original = ((x >= x_min) & (x <= x_max) &
                               (y >= y_min) & (y <= y_max) &
                               (z >= z_min) & (z <= z_max)).sum()
        distance_original = np.sqrt(x_center ** 2 + y_center ** 2 + z_center ** 2)
        results_original.append({'File_Name': file_name, 'Time': time, 'Num_Points': num_points_original, 'Distance': distance_original})

        # Ground Plane Cuboid
        x_min, x_max = x_center_shifted - length_shifted / 2, x_center_shifted + length_shifted / 2
        y_min, y_max = y_center - width_shifted / 2, y_center + width_shifted / 2
        z_min, z_max = z_center_shifted - height_shifted / 2, z_center_shifted + height_shifted / 2
        num_points_shifted = ((x >= x_min) & (x <= x_max) &
                              (y >= y_min) & (y <= y_max) &
                              (z >= z_min) & (z <= z_max)).sum()
        distance_shifted = np.sqrt(x_center_shifted ** 2 + y_center ** 2 + z_center_shifted ** 2)
        results_shifted.append({'File_Name': file_name, 'Time': time, 'Num_Points': num_points_shifted, 'Distance': distance_shifted})

    df_original = pd.DataFrame(results_original)
    df_shifted = pd.DataFrame(results_shifted)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_original['Distance'], df_original['Num_Points'], marker='o', linestyle='-', color='blue')
    # ax1.set_title('Points within Object Cuboid')
    ax1.set_xlabel('Distance to Object From FSN (m)', fontsize=22)
    ax1.set_ylabel('Number of Points in Detection', fontsize=22)
    formatter = FuncFormatter(lambda x, _: f'{x:.0f}')
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(50))  # Adjust major ticks
    ax1.yaxis.set_minor_locator(MultipleLocator(10))  # Adjust minor ticks
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
    ax1.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)
    plt.tight_layout()
    # plt.savefig(
    #     'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
    #     'scenario_3_baseline/sc3_node_object_cuboid_points_2.png',
    #     bbox_inches='tight', dpi=400, format='png')

    plt.savefig(
        'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P0_Thesis/Images/'
        'sc3_node_object_cuboid_points_2.png',
        bbox_inches='tight', dpi=400, format='png')



    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(df_shifted['Distance'], df_shifted['Num_Points'], marker='o', linestyle='-', color='red')
    # ax2.set_title('1x1x1 Cuboid intersecting the LiDAR ground plane.')
    ax2.set_xlabel('Distance to Object (m)', fontsize=22)
    ax1.set_ylabel('Number of Points in Cuboid', fontsize=22)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax2.yaxis.set_major_locator(MultipleLocator(20))  # Adjust major ticks
    ax2.yaxis.set_minor_locator(MultipleLocator(5))  # Adjust minor ticks
    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
    ax2.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18, rotation=0)
    plt.tight_layout()
    plt.savefig(
        'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
        'scenario_3_baseline/sc3_node_ground_plane_cuboid_points_2.png',
        bbox_inches='tight', dpi=400, format='png')
    # plt.show()

    return None

def car_annotation_quality():
    car_data_directory = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/car_cepton_overlap_1/person_1/out/'
    car_annotation_csv = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-755_id-904_fixed_person_1_annotations_car.csv'

    car_annotations = pd.read_csv(car_annotation_csv)
    results_original = []
    results_shifted = []

    for _, row in car_annotations.iterrows():
        file_name = row['File_Name']
        time = row['Time_Short']

        # ORIGINAL CUBOID
        length, width, height = row['Length'], row['Width'], row['Height']
        x_center, y_center, z_center = row['X_Center'], row['Y_Center'], row['Z_Center']

        if length <= 0 or width <= 0 or height <= 0 or (x_center == 0 and y_center == 0 and z_center == 0):
            results_original.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            results_shifted.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            print(f"Skipped invalid cuboid in file: {file_name}")
            continue

        las_path = os.path.join(car_data_directory, file_name)
        if not os.path.exists(las_path):
            print(f".las file not found: {las_path}")
            results_original.append({'File_Name': file_name, 'Time': time, 'Num_Points': 0, 'Distance': 0})
            continue

        las = pylas.read(las_path)
        x = las.X * las.header.scales[0] + las.header.offsets[0]
        y = las.Y * las.header.scales[1] + las.header.offsets[1]
        z = las.Z * las.header.scales[2] + las.header.offsets[2]

        # Original Cuboid
        x_min, x_max = x_center - length / 2, x_center + length / 2
        y_min, y_max = y_center - width / 2, y_center + width / 2
        z_min, z_max = z_center - height / 2, z_center + height / 2
        num_points_original = ((x >= x_min) & (x <= x_max) &
                               (y >= y_min) & (y <= y_max) &
                               (z >= z_min) & (z <= z_max)).sum()
        distance_original = np.sqrt(x_center ** 2 + y_center ** 2 + z_center ** 2)
        results_original.append({'File_Name': file_name, 'Time': time, 'Num_Points': num_points_original, 'Distance': distance_original})


    df_original = pd.DataFrame(results_original)
    df_original = df_original[df_original['Num_Points'] != 0]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_original['Distance'], df_original['Num_Points'], marker='o', linestyle='-', color='blue')
    # ax1.set_title('Points within Object Cuboid')
    ax1.set_xlabel('Distance to Object (m)', fontsize=22)
    ax1.set_ylabel('Number of Points in Detection', fontsize=22)
    formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(MultipleLocator(5))  # Adjust major ticks
    ax1.yaxis.set_minor_locator(MultipleLocator(1))  # Adjust minor ticks
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
    ax1.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    plt.xticks(fontsize=22, rotation=45)
    plt.yticks(fontsize=22, rotation=0)
    plt.tight_layout()
    plt.savefig(
        'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
        'scenario_3_baseline/sc3_car_object_cuboid_points.png',
        bbox_inches='tight', dpi=400, format='png')

    # plt.show()


    return None

def plot_baseline(delay_pkl, vehicle_annotations, node_annotations):
    fig, ax = plt.subplots(figsize=(10, 6))

    def format_tick_value_x(value, pos):
        return f'{value:.0f}'

    def format_tick_value_y(value, pos):
        return f'{value:.2f}'

    formatter_x = FuncFormatter(format_tick_value_x)
    formatter_y = FuncFormatter(format_tick_value_y)
    small_threshold = 1e-6
    eucl_dist_df = delay_pkl['eucl_dist_dict'][0] # Baseline - no delay.
    filtered_eucl_dist_df = eucl_dist_df[(np.abs(eucl_dist_df['car_base_to_car_obj']) > small_threshold) &
                                        (np.abs(eucl_dist_df['node_base_to_node_obj']) > small_threshold) &
                                        (~eucl_dist_df['car_base_to_car_obj'].isna()) &
                                        (~eucl_dist_df['node_base_to_node_obj'].isna())]

    node_base_to_node_obj = filtered_eucl_dist_df['node_base_to_node_obj']
    car_obj_to_node_obj = filtered_eucl_dist_df['car_obj_to_node_obj']

    ax.plot(node_base_to_node_obj, car_obj_to_node_obj, label='Baseline', marker='o')
    # ax.set_title('System Baseline vs Distance', fontsize=22)
    ax.set_ylabel('Euclidean Distance (m)', fontsize=22)
    ax.set_xlabel('Distance to Object From FSN (m)', fontsize=22)
    ax.xaxis.set_major_locator(MultipleLocator(1)) # Major Tick every 1 meter.
    ax.xaxis.set_minor_locator(MultipleLocator(0.2)) # Minor Tick every 0.2 meter.
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Major Tick every 0.1 meter.
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # Minor Tick every 0.01 meter.
    ax.xaxis.set_major_formatter(formatter_x)  # Apply 2 decimal places to x-axis
    ax.yaxis.set_major_formatter(formatter_y)  # Apply 2 decimal places to y-axis
    ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)
    plt.tight_layout()
    # plt.savefig('C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/scenario_3_baseline/sc3_error_vs_distance.png',
    #             bbox_inches='tight', dpi=400, format='png')

    plt.savefig('C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P0_Thesis/Images/sc3_error_vs_distance.png',
        bbox_inches='tight', dpi=400, format='png')
    plt.show()



    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 90), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated= rotate_all_coordinates(
        np.array(all_x_centers_car), np.array(all_y_centers_car), np.array(all_z_centers_car), R_lidar_to_local)

    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car, all_x_centers_node, all_y_centers_node,
                   all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=25)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_baseline.csv"


    plot_global_map(geotiff_path, global_baseline_csv)
    # node_point_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/car_cepton_overlap_1/person_1/out/"
    # car_point_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/car_cepton_overlap_1/person_1/out"
    # compass_list = [35.77, 35.89, 36.05, 36.14, 36.27, 36.82, 36.88]
    # compass_heading = compass_list[0] - 15  # 15 degrees offset since the RT from car to node was aligned with 15 degrees rotated car point cloud.
    # geo_transformer = GeoTransformer()
    # node_gps = (53.29107151901461, -9.070670083320563)
    # car_gps = (53.290474634336654, -9.071039198388549, 0)
    #
    # df_name_R = pd.read_csv(
    #     'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_with_R_for_local_to_ecef.csv')
    # df_name_t_int = pd.read_csv(
    #     'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/local_global_rotation_Tn/x_y_gps_pair_names_for_local_to_ecef_w_SLAM_T0-Tn_Interval.csv')
    # # Merged R matrix, and time interval from df_name_R and df_name_t_int.
    # df_R_t_interval = pd.merge(df_name_R, df_name_t_int, on='Base_Point')
    #
    # output_csv_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/car_cepton_overlap_1/person_1/out/baseline_points_pcd.csv"
    #
    # plot_point_global(geo_transformer, geotiff_path, output_csv_path, node_gps, node_annotations, node_point_path, compass_heading, df_R_t_interval, car_gps, vehicle_annotations, car_point_path)

def plot_delay(delay_pkl, vehicle_annotations, node_annotations):
    fig, ax = plt.subplots()

    relative_error_diff_df = delay_pkl['relative_error_diff_df']
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m', '14-15m', '15-16m', '16-17m', '17-18m']
    delay_range = np.arange(0, 1000, 100)

    # Extract difference columns
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
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
    for _, row in relative_error_diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in diff_columns])

        # Determine the range label for this row and group data
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(diff_values)

    # Add range averages to the plot
    for range_label, values in range_averages.items():
        avg_values = np.nanmean(values, axis=0)  # Compute the mean across rows
        plt.plot(delays, avg_values, marker='o', label=f'{range_label}')

    # ax.set_title('Impact of Temporal Delay on Spatial Uncertainty', fontsize=16)
    ax.set_xlabel('Delay (ms)', fontsize=30)
    ax.set_ylabel('Euclidean Distance (m)', fontsize=30)

    # Major and minor ticks for the x-axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    formatter = ticker.FormatStrFormatter('%.2f')
    ax.yaxis.set_major_formatter(formatter)

    # Grid settings
    ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines

    ax.legend(title='Averaged Distance Ranges', title_fontsize=17, fontsize=17, loc='upper left')
    plt.tight_layout()
    plt.xticks(fontsize=30, rotation=45)
    plt.yticks(fontsize=30, rotation=0)
    plt.tight_layout()
    plt.savefig(
        'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results'
        '/scenario_3_delay/sc3_error_vs_delay_avg.png',
        bbox_inches='tight', dpi=400, format='png')

    plt.show()

    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 10), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated = rotate_all_coordinates(np.array(all_x_centers_car), np.array(all_y_centers_car),
                                                                                                 np.array(all_z_centers_car), R_lidar_to_local)

    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car, all_x_centers_node, all_y_centers_node,
                   all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=10)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_delay.csv"
    plot_delay_global_map(geotiff_path, global_baseline_csv)

def plot_heading(heading_pkl):
    relative_error_diff_df = heading_pkl['relative_error_diff_df']
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    heading_ranges = {
        "Full Range": np.round(np.arange(-2.0, 2.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-2.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 2.1, 0.1), 2)
    }

    # Compass list and its adjustments
    compass_list = [35.77, 35.89, 36.05, 36.14, 36.27, 36.82, 36.88]
    compass_heading = compass_list[0] - 15

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    heading_pc_err_list = []

    for col in diff_columns:
        try:
            heading_pc_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            heading_pc_err_list.append(heading_pc_err)
        except ValueError:
            pass

    for range_name, heading_range in heading_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(heading_pc_err_list) if h_err in heading_range]
        heading_pc_errs = [heading_pc_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        heading_node = [compass_heading + h_err for h_err in heading_pc_errs]

        # Compute range averages
        range_averages = defaultdict(list)
        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        # Create a new plot for each range
        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(heading_pc_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of Node Heading Error: {range_name}', fontsize=16)
        ax.set_xlabel('GPS Compass Heading Error ($^\circ$)', fontsize=22)
        ax.set_ylabel('Euclidean Distance (m)', fontsize=22)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

        formatter = ticker.FormatStrFormatter('%.1f')
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=18)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=17)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14)
        plt.tight_layout()
        # plt.xticks(fontsize=17, rotation=90)
        plt.yticks(fontsize=17, rotation=0)
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_heading/sc3_error_vs_heading_avg.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    geotiff_path = 'C:/Users/Roshan George/Desktop/P2_Misc/WP2/danganGeoTiff.tif'
    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_heading_error.csv"
    plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title="Node Heading Error")

def plot_x_rot(x_rot_pkl, vehicle_annotations, node_annotations):
    relative_error_diff_df = x_rot_pkl['relative_error_diff_df']
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    x_rot_ranges = {
        "Full Range": np.round(np.arange(-5.0, 5.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-5.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 5.1, 0.1), 2)
    }

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    x_rot_err_list = []

    for col in diff_columns:
        try:
            x_rot_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            x_rot_err_list.append(x_rot_err)
        except ValueError:
            pass

    for range_name, heading_range in x_rot_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(x_rot_err_list) if h_err in heading_range]
        x_rot_errs = [x_rot_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        # Compute range averages
        range_averages = defaultdict(list)

        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(x_rot_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of X-Axis Rotation Error on Spatial Uncertainty: {range_name}', fontsize=16)
        ax.set_xlabel('X-Axis Rotation ($^\circ$)', fontsize=22)
        ax.set_ylabel('Euclidean Distance(m)', fontsize=22)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        formatter = ticker.FormatStrFormatter('%.1f')
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=18)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=16)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(0, 1.4)

        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14)
        plt.tight_layout()
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16, rotation=0)
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_x_rotation/sc3_error_vs_xrot_avg.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 10), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated = rotate_all_coordinates(np.array(all_x_centers_car), np.array(all_y_centers_car), np.array(all_z_centers_car), R_lidar_to_local)
    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car, all_x_centers_node,
                   all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=10)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_x_rotation.csv"
    plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title="X Rotation Error")

def plot_y_rot(y_rot_pkl, vehicle_annotations, node_annotations):
    relative_error_diff_df = y_rot_pkl['relative_error_diff_df']
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    rot_ranges = {
        "Full Range": np.round(np.arange(-5.0, 5.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-5.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 5.1, 0.1), 2)
    }

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    rot_err_list = []

    for col in diff_columns:
        try:
            rot_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            rot_err_list.append(rot_err)

        except ValueError:
            pass

    for range_name, heading_range in rot_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(rot_err_list) if h_err in heading_range]
        x_rot_errs = [rot_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        # Compute range averages
        range_averages = defaultdict(list)

        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(x_rot_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of Y-Axis Rotation Error on Spatial Uncertainty: {range_name}', fontsize=16)
        ax.set_xlabel('Y-Axis Rotation ($^\circ$)', fontsize=18)
        ax.set_ylabel('Euclidean Distance(m)', fontsize=18)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        formatter = ticker.FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=16)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=16)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=16)
        plt.setp(ax.yaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=16)

        # formatter = ticker.FormatStrFormatter('%.2f')
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)

        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14, loc='upper left')
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_y_rotation/sc3_error_vs_yrot_avg_3.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 10), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated = rotate_all_coordinates(np.array(all_x_centers_car), np.array(all_y_centers_car),
                                                                                                 np.array(all_z_centers_car), R_lidar_to_local)

    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car,
                   all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=10)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_y_rotation.csv"
    plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title="Y Rotation Error")

def plot_z_rot(z_rot_pkl, vehicle_annotations, node_annotations):
    relative_error_diff_df = z_rot_pkl['relative_error_diff_df']
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    rot_ranges = {
        "Full Range": np.round(np.arange(-5.0, 5.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-5.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 5.1, 0.1), 2)
    }

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    rot_err_list = []

    for col in diff_columns:
        try:
            rot_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            rot_err_list.append(rot_err)

        except ValueError:
            pass

    for range_name, heading_range in rot_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(rot_err_list) if h_err in heading_range]
        x_rot_errs = [rot_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        # Compute range averages
        range_averages = defaultdict(list)

        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(x_rot_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of Z-Axis Rotation Error on Spatial Uncertainty: {range_name}', fontsize=16)
        ax.set_xlabel('Z-Axis Rotation ($^\circ$)', fontsize=18)
        ax.set_ylabel('Euclidean Distance(m)', fontsize=18)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # formatter = ticker.FormatStrFormatter('%.2f')
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)

        formatter = ticker.FormatStrFormatter('%.1f')
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=18)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=16)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha="right", fontsize=16)
        plt.setp(ax.yaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=16)
        ax.yaxis.set_major_formatter(formatter)


        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_z_rotation/sc3_error_vs_zrot_avg_3.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 10), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated = rotate_all_coordinates(np.array(all_x_centers_car), np.array(all_y_centers_car),
                                                                                                 np.array(all_z_centers_car), R_lidar_to_local)

    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car,
                   all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=10)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_z_rotation.csv"
    plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title="Z Rotation Error")

def plot_x_tran(x_tran_pkl, vehicle_annotations, node_annotations):
    relative_error_diff_df = x_tran_pkl['relative_error_diff_df']
    time_correction_df = x_tran_pkl['eucl_dist_dict'][-0.0]

    relative_error_diff_df = relative_error_diff_df.merge(
        time_correction_df[['Time_Car', 'node_base_to_node_obj']],
        on='Time_Car',
        suffixes=('', '_corr')
    )

    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    rot_ranges = {
        "Full Range": np.round(np.arange(-5.0, 5.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-5.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 5.1, 0.1), 2)
    }

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    rot_err_list = []

    for col in diff_columns:
        try:
            rot_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            rot_err_list.append(rot_err)

        except ValueError:
            pass

    for range_name, heading_range in rot_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(rot_err_list) if h_err in heading_range]
        x_rot_errs = [rot_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        # Compute range averages
        range_averages = defaultdict(list)

        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj_corr']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(x_rot_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of X-Axis Translation Error on Spatial Uncertainty: {range_name}', fontsize=16)
        ax.set_xlabel('X-Axis Translation (m)', fontsize=18)
        ax.set_ylabel('Euclidean Distance (m)', fontsize=18)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # formatter = ticker.FormatStrFormatter('%.2f')
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)

        formatter = ticker.FormatStrFormatter('%.1f')
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=18)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha="right", fontsize=18)
        plt.setp(ax.yaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_x_translation/sc3_error_vs_xtran_avg_3.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 10), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated = rotate_all_coordinates(np.array(all_x_centers_car), np.array(all_y_centers_car),
                                                                                                 np.array(all_z_centers_car), R_lidar_to_local)

    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car,
                   all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=10)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_x_translation.csv"
    plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title="X Translation Error")

def plot_y_tran(y_tran_pkl, vehicle_annotations, node_annotations):
    relative_error_diff_df = y_tran_pkl['relative_error_diff_df']
    time_correction_df = y_tran_pkl['eucl_dist_dict'][-0.0]

    relative_error_diff_df = relative_error_diff_df.merge(
        time_correction_df[['Time_Car', 'node_base_to_node_obj']],
        on='Time_Car',
        suffixes=('', '_corr')
    )


    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    rot_ranges = {
        "Full Range": np.round(np.arange(-5.0, 5.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-5.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 5.1, 0.1), 2)
    }

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    rot_err_list = []

    for col in diff_columns:
        try:
            rot_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            rot_err_list.append(rot_err)

        except ValueError:
            pass

    for range_name, heading_range in rot_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(rot_err_list) if h_err in heading_range]
        x_rot_errs = [rot_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        # Compute range averages
        range_averages = defaultdict(list)

        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj_corr']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(x_rot_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of Y-Axis Translation Error on Spatial Uncertainty: {range_name}', fontsize=16)
        ax.set_xlabel('Y-Axis Translation (m)', fontsize=18)
        ax.set_ylabel('Euclidean Distance (m)', fontsize=18)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # formatter = ticker.FormatStrFormatter('%.2f')
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)

        formatter = ticker.FormatStrFormatter('%.1f')
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=18)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha="right", fontsize=18)
        plt.setp(ax.yaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_y_translation/sc3_error_vs_ytran_avg_3.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

    data_car = pd.read_csv(vehicle_annotations)
    data_node = pd.read_csv(node_annotations)

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]

    all_x_centers_car, all_y_centers_car, all_z_centers_car, all_widths_car, all_lengths_car = [], [], [], [], []
    all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node = [], [], [], [], []

    for i in tqdm(range(0, 10), desc="Processing rows"):
        row_car = data_car.iloc[i]
        row_node = data_node.iloc[i]
        all_x_centers_car.append(row_car['X_Center'])
        all_y_centers_car.append(row_car['Y_Center'])
        all_z_centers_car.append(row_car['Z_Center'])
        all_widths_car.append(row_car['Width'])
        all_lengths_car.append(row_car['Length'])
        all_x_centers_node.append(row_node['X_Center'])
        all_y_centers_node.append(row_node['Y_Center'])
        all_z_centers_node.append(row_node['Z_Center'])
        all_widths_node.append(row_node['Width'])
        all_lengths_node.append(row_node['Length'])

    x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated = rotate_all_coordinates(np.array(all_x_centers_car), np.array(all_y_centers_car),
                                                                                                 np.array(all_z_centers_car), R_lidar_to_local)

    plot_local_map(x_centers_car_rotated, y_centers_car_rotated, z_centers_car_rotated, all_widths_car, all_lengths_car,
                   all_x_centers_node, all_y_centers_node, all_z_centers_node, all_widths_node, all_lengths_node, max_plot_range=10)

    geotiff_path = 'C:/Users/Roshan George/Desktop/Geotiff_3/DJI_0386_modified.tif'
    global_baseline_csv = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_y_translation.csv"
    # plot_heading_rotation_global_map_2(geotiff_path, global_baseline_csv, label_title="Y Translation Error") # use for virdis
    plot_heading_rotation_global_map(geotiff_path, global_baseline_csv, label_title="Y Translation Error")

def plot_z_tran(z_tran_pkl, vehicle_annotations, node_annotations):
    relative_error_diff_df = z_tran_pkl['relative_error_diff_df']
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m',
                    '14-15m', '15-16m', '16-17m', '17-18m']

    # Define the three heading ranges
    rot_ranges = {
        "Full Range": np.round(np.arange(-5.0, 5.1, 0.1), 2),
        "Negative Range": np.round(np.arange(-5.0, 0.1, 0.1), 2),
        "Positive Range": np.round(np.arange(0.0, 5.1, 0.1), 2)
    }

    # Extract columns representing heading errors
    diff_columns = [col for col in relative_error_diff_df.columns if col.startswith('Diff_')]
    rot_err_list = []

    for col in diff_columns:
        try:
            rot_err = float(col.split('_')[1].split('-', maxsplit=1)[1])
            rot_err_list.append(rot_err)

        except ValueError:
            pass

    for range_name, heading_range in rot_ranges.items():
        # Filter data based on the heading range
        filtered_indices = [i for i, h_err in enumerate(rot_err_list) if h_err in heading_range]
        x_rot_errs = [rot_err_list[i] for i in filtered_indices]
        filtered_diff_columns = [diff_columns[i] for i in filtered_indices]

        # Compute range averages
        range_averages = defaultdict(list)

        for _, row in relative_error_diff_df.iterrows():
            node_base_to_node_obj = row['node_base_to_node_obj']
            diff_values = np.array([row[col] for col in filtered_diff_columns])

            for i, (lower, upper) in enumerate(distance_ranges):
                if lower <= node_base_to_node_obj < upper:
                    range_averages[range_labels[i]].append(diff_values)

        fig, ax = plt.subplots(figsize=(10, 6))

        for range_label, values in range_averages.items():
            avg_values = np.nanmean(values, axis=0)
            plt.plot(x_rot_errs, avg_values, marker='o', label=f'{range_label}')

        # ax.set_title(f'Impact of Z-Axis Translation Error on Spatial Uncertainty: {range_name}', fontsize=16)
        ax.set_xlabel('Z-Axis Translation (m)', fontsize=18)
        ax.set_ylabel('Euclidean Distance (m)', fontsize=18)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        formatter = ticker.FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        formatter = ticker.FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=18)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=18)
        plt.setp(ax.yaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=18)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax.legend(title='Averaged Distance Ranges', fontsize=14, loc='upper left')
        plt.tight_layout()
        plt.savefig(
            'C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/results/'
            'scenario_3_z_translation/sc3_error_vs_ztran_avg_1.png',
            bbox_inches='tight', dpi=400, format='png')
        plt.show()

def plot_bbox_uncertainty():
    # node_annotation_csv = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_person_1_annotations_node.csv'
    node_annotation_csv = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-755_id-904_fixed_person_1_annotations_car.csv'
    node_annotations = pd.read_csv(node_annotation_csv)

    # Plot X_Center and Y_Center over time
    plt.figure(figsize=(10, 6))
    plt.plot(node_annotations["Time_Short"], node_annotations["X_Center"], label="X Center", marker="o",linestyle="--", alpha=0.7)
    # plt.plot(node_annotations["Time_Short"], node_annotations["Y_Center"], label="Y Center", marker="s",linestyle="--", alpha=0.7)

    # Formatting
    plt.xlabel("Time (s)")
    plt.ylabel("Bounding Box Center Coordinates (m)")
    plt.title("Bounding Box Center Variation Over Time (Filtered)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return None

def plot_sim_delay_offset(log_scale=False):
    # Given speed values in km/h
    speeds_kmh = [30, 50, 60, 80, 100, 120]

    # Convert speeds from km/h to m/s
    speeds_ms = [kmh / 3.6 for kmh in speeds_kmh]

    # Convert speeds from m/s to mm/ms (meters per millisecond)
    speeds_mms = [ms / 1000 for ms in speeds_ms]

    # Define the transmission delays in milliseconds (from 100ms to 1000ms in 100ms increments)
    delays_ms = list(range(100, 1001, 10))

    # Compute positional offsets for each delay and speed
    positional_offsets = {delay: [speed_mms * delay for speed_mms in speeds_mms] for delay in delays_ms}

    # Convert dictionary to DataFrame for easier handling
    df = pd.DataFrame(positional_offsets, index=speeds_kmh).T  # Transpose to make delay the x-axis
    df.columns = [f"{speed} km/h" for speed in speeds_kmh]  # Rename columns for legend

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    for speed in df.columns:
        ax.plot(df.index, df[speed], marker='o', label=speed)

    ax.set_xlabel('Transmission Delay (ms)', fontsize=22)
    ax.set_ylabel('Positional Offset (m)', fontsize=22)

    if log_scale:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())


    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Set formatter for major ticks
    formatter = ticker.FormatStrFormatter('%.2f')
    ax.yaxis.set_major_formatter(formatter)

    # 🔥 Remove labels for minor ticks
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Adjust tick labels formatting
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="center", fontsize=17)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha="right", fontsize=17)

    # Grid settings
    ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')

    plt.legend(title="Object Speed", title_fontsize=17, fontsize=15, loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    vehicle_annotations = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-755_id-904_fixed_person_1_annotations_car.csv"
    node_annotations = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_fixed_person_1_annotations_node.csv"
    time_validation = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/time_validation.csv"

    with open('pickles/scenario_3_exp_1_3.pkl', 'rb') as f:
        delay_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_2_3.pkl', 'rb') as f:
        heading_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_3_x_1.pkl', 'rb') as f:
        x_rot_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_3_y_2.pkl', 'rb') as f:
        y_rot_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_3_z_3.pkl', 'rb') as f:
        z_rot_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_4_x_t_1.pkl', 'rb') as f:
        x_tran_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_4_y_t_1.pkl', 'rb') as f:
        y_tran_pkl = pickle.load(f)

    with open('pickles/scenario_3_exp_4_z_t_1.pkl', 'rb') as f:
        z_tran_pkl = pickle.load(f)

    # Dataset location for tif has changed into the images directory in onedrive

    # plot_csv_data(time_validation, range_name='My Data Range', label='Difference Data')
    # plot_baseline(delay_pkl, vehicle_annotations, node_annotations)
    node_annotation_quality()
    # car_annotation_quality()

    # plot_delay(delay_pkl, vehicle_annotations, node_annotations)
    # plot_heading(heading_pkl)
    # plot_x_rot(x_rot_pkl, vehicle_annotations, node_annotations)
    # plot_y_rot(y_rot_pkl, vehicle_annotations, node_annotations)
    # plot_z_rot(z_rot_pkl, vehicle_annotations, node_annotations)
    # plot_x_tran(x_tran_pkl, vehicle_annotations, node_annotations)
    # plot_y_tran(y_tran_pkl, vehicle_annotations, node_annotations)
    # plot_z_tran(z_tran_pkl, vehicle_annotations, node_annotations)

    # plot_bbox_uncertainty()
    # plot_sim_delay_offset(log_scale=False)

if __name__ == "__main__":
    main()

