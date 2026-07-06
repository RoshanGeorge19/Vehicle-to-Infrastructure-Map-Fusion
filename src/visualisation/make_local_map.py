import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

# # Function to plot radar-like concentric circles
# def add_radar_guidelines(ax, max_range, num_circles=6):
#     step_size = max_range / num_circles
#     step_size = max(int(np.ceil(step_size)), 1)  # Ensure at least step size of 1
#
#     angles = np.linspace(0, 2 * np.pi, 360)
#     radii = np.arange(0, max_range + step_size, step_size)
#
#     # Draw concentric circles
#     for r in radii:
#         ax.plot(r * np.cos(angles), r * np.sin(angles), color='gray', lw=0.5, linestyle="--")
#         # Add labels where circles intersect axes
#         if r != 0:  # Exclude origin
#             ax.text(r, 0.2, f"{int(r)}", color='white', fontsize=13, ha='center', va='center')  # X+ axis
#             ax.text(-r, 0.2, f"-{int(r)}", color='white', fontsize=13, ha='center', va='center')  # X- axis
#             ax.text(0.25, r, f"{int(r)}", color='white', fontsize=13, ha='center', va='center')  # Y+ axis
#             ax.text(0.25, -r, f"-{int(r)}", color='white', fontsize=13, ha='center', va='center')  # Y- axis
#
#     # Draw crosshairs
#     ax.plot([-max_range, max_range], [0, 0], color='white', lw=0.7)  # Horizontal
#     ax.plot([0, 0], [-max_range, max_range], color='white', lw=0.7)  # Vertical
#
#     # Add grid lines
#     ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
#
#     # Set consistent tick intervals
#     step_size = max_range // num_circles
#     if step_size == 0:
#         step_size = 1  # Ensure a valid step size
#     ticks = np.arange(-max_range, max_range + 1, step_size)  # Create uniform ticks
#
#     if 0 in ticks:  # Remove the origin (0)
#         ticks = ticks[ticks != 0]
#     ax.set_xticks(ticks)  # Set consistent X-axis ticks
#     ax.set_yticks(ticks)  # Set consistent Y-axis ticks
#
# # Function to plot bounding boxes
# def add_bounding_boxes(ax, x_centers, y_centers, widths, lengths):
#     for x_center, y_center, width, length in zip(x_centers, y_centers, widths, lengths):
#         rect_x = x_center - width / 2
#         rect_y = y_center - length / 2
#         rect = plt.Rectangle(
#             (rect_x, rect_y), width, length,
#             edgecolor='cyan', facecolor='none', lw=1
#         )
#         ax.add_patch(rect)
#
#     # Highlight the center point (optional)
#     ax.scatter(x_centers, y_centers, color='red', s=15, label="Centers")
#
# # Main function for plotting lidar annotations in subplots
# def plot_lidar_annotations_subplots(x_centers_car, y_centers_car, widths_car, lengths_car,
#                                     x_centers_node, y_centers_node, widths_node, lengths_node, filename):
#
#     # Create a figure with 1 row and 2 columns for subplots
#     fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
#     fig.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots
#
#     # Calculate max ranges for both datasets
#     max_range_car = max(max(np.abs(x_centers_car)) + max(widths_car) / 2, max(np.abs(y_centers_car)) + max(lengths_car) / 2)
#     max_range_node = max(max(np.abs(x_centers_node)) + max(widths_node) / 2, max(np.abs(y_centers_node)) + max(lengths_node) / 2)
#
#     # Use the global max range to synchronize axis limits
#     # global_max_range = max(max_range_car, max_range_node) + 5  # Add buffer space
#     global_max_range = 25  # Add buffer space
#
#     # Generate equal ticks based on the global max range
#     num_circles = 6  # Number of concentric circles
#     step_size = global_max_range / num_circles
#     step_size = max(int(np.ceil(step_size)), 1)  # Ensure step size is at least 1
#
#     ticks = np.arange(-global_max_range, global_max_range + step_size, step_size)
#
#     # Plot for the "Car" dataset (subplot 1)
#     ax1 = axs[0]
#     ax1.set_facecolor('black')
#     add_radar_guidelines(ax1, global_max_range)
#     add_bounding_boxes(ax1, x_centers_car, y_centers_car, widths_car, lengths_car)
#     ax1.set_xlim(-global_max_range, global_max_range)
#     ax1.set_ylim(-global_max_range, global_max_range)
#     ax1.set_aspect('equal', adjustable='box')
#     ax1.set_title("Car - Local Map", fontsize=18, color='white', pad=20)
#
#     # Apply synchronized ticks for the "Car" dataset
#     ax1.set_xticks(ticks)
#     ax1.set_yticks(ticks)
#
#     # Plot for the "Node" dataset (subplot 2)
#     ax2 = axs[1]
#     ax2.set_facecolor('black')
#     add_radar_guidelines(ax2, global_max_range)
#     add_bounding_boxes(ax2, x_centers_node, y_centers_node, widths_node, lengths_node)
#     ax2.set_xlim(-global_max_range, global_max_range)
#     ax2.set_ylim(-global_max_range, global_max_range)
#     ax2.set_aspect('equal', adjustable='box')
#     ax2.set_title("Node - Local Map", fontsize=18, color='white', pad=20)
#
#     # Apply synchronized ticks for the "Node" dataset
#     ax2.set_xticks(ticks)
#     ax2.set_yticks(ticks)
#
#     # Display the plot
#     # plt.tight_layout()
#     plt.show()
#
#     # Save the plot as an image file
#     plt.tight_layout()
#     # plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save as high-quality PNG
#     plt.close(fig)  # Close the figure to free up memory
#
# def plot_lidar_annotations(x_centers_car, y_centers_car, widths_car, lengths_car,
#                            x_centers_node, y_centers_node, widths_node, lengths_node,
#                            filename_car):
#
#     # Define a fixed global range for both plots
#     global_max_range = 10  # Adjust range as needed
#     num_circles = 5  # Number of concentric circles
#     step_size = global_max_range / num_circles
#     step_size = max(int(np.ceil(step_size)), 1)  # Ensure a minimum step size of 1
#     ticks = np.arange(-global_max_range, global_max_range + step_size, step_size)
#
#     # --- PLOT 1: Car Dataset ---
#     fig_car, ax_car = plt.subplots(figsize=(8, 8), facecolor='black')
#     ax_car.set_facecolor('black')
#     add_radar_guidelines(ax_car, global_max_range)
#     add_bounding_boxes(ax_car, x_centers_car, y_centers_car, widths_car, lengths_car)
#     ax_car.set_xlim(-global_max_range, global_max_range)
#     ax_car.set_ylim(-global_max_range, global_max_range)
#     ax_car.set_aspect('equal', adjustable='box')
#     ax_car.set_title("Car - Local Map", fontsize=18, color='white', pad=20)
#
#     # Apply ticks
#     ax_car.set_xticks(ticks)
#     ax_car.set_yticks(ticks)
#
#     # Show & save car plot
#     plt.show()
#     plt.tight_layout()
#     # plt.savefig(filename_car, dpi=300, bbox_inches='tight')
#     plt.close(fig_car)
#
#     # --- PLOT 2: Node Dataset ---
#     fig_node, ax_node = plt.subplots(figsize=(8, 8), facecolor='black')
#     ax_node.set_facecolor('black')
#     add_radar_guidelines(ax_node, global_max_range)
#     add_bounding_boxes(ax_node, x_centers_node, y_centers_node, widths_node, lengths_node)
#     ax_node.set_xlim(-global_max_range, global_max_range)
#     ax_node.set_ylim(-global_max_range, global_max_range)
#     ax_node.set_aspect('equal', adjustable='box')
#     ax_node.set_title("Node - Local Map", fontsize=18, color='white', pad=20)
#
#     # Apply ticks
#     ax_node.set_xticks(ticks)
#     ax_node.set_yticks(ticks)
#
#     # Show & save node plot
#     plt.show()
#     plt.tight_layout()
#     # plt.savefig(filename_node, dpi=300, bbox_inches='tight')
#     plt.close(fig_node)

def add_radar_guidelines(ax, max_range, num_circles=6):
    """ Draws concentric circles and grid lines for visualization """
    step_size = max_range / num_circles
    step_size = max(int(np.ceil(step_size)), 1)  # Ensure at least step size of 1

    angles = np.linspace(0, 2 * np.pi, 360)
    radii = np.arange(0, max_range + step_size, step_size)

    for r in radii:
        ax.plot(r * np.cos(angles), r * np.sin(angles), color='gray', lw=0.5, linestyle="--")
        if r != 0:  # Exclude origin
            ax.text(r, 0.2, f"{int(r)}", color='white', fontsize=13, ha='center', va='center')
            ax.text(-r, 0.2, f"-{int(r)}", color='white', fontsize=13, ha='center', va='center')
            ax.text(0.25, r, f"{int(r)}", color='white', fontsize=13, ha='center', va='center')
            ax.text(0.25, -r, f"-{int(r)}", color='white', fontsize=13, ha='center', va='center')

    ax.plot([-max_range, max_range], [0, 0], color='white', lw=0.7)
    ax.plot([0, 0], [-max_range, max_range], color='white', lw=0.7)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ticks = np.arange(-max_range, max_range + 1, step_size)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def rotate_bounding_boxes(x_centers, y_centers, widths, lengths, rotations_z):
    """ Rotates bounding boxes around their centers using their respective angles. """
    rotated_boxes = []

    for x_center, y_center, width, length, theta_deg in zip(x_centers, y_centers, widths, lengths, rotations_z):
        theta_deg = theta_deg - 90  # Adjust for the difference in coordinate systems
        theta_rad = np.deg2rad(theta_deg)  # Convert degree to radians

        # Rotation Matrix
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])

        half_w, half_l = width / 2, length / 2
        corners = np.array([
            [-half_w, -half_l], [half_w, -half_l],
            [half_w, half_l], [-half_w, half_l]
        ])

        # Rotate corners
        rotated_corners = np.dot(R, corners.T).T

        # Shift back to global position
        rotated_corners[:, 0] += x_center
        rotated_corners[:, 1] += y_center

        rotated_boxes.append(rotated_corners)

    return rotated_boxes


def add_rotated_bounding_boxes(ax, rotated_boxes):
    """ Plots rotated rectangular bounding boxes """
    for corners in rotated_boxes:
        closed_corners = np.vstack([corners, corners[0]])
        ax.plot(closed_corners[:, 0], closed_corners[:, 1], color="cyan", linewidth=1.2)
        center_x, center_y = np.mean(corners, axis=0)
        ax.scatter(center_x, center_y, color='red', s=15, label="Centers")


def rotate_all_coordinates(x_coords, y_coords, rotation_matrix):
    """ Rotates a list of x-y coordinates using a 3x3 rotation matrix """
    homogeneous_coords = np.vstack((x_coords, y_coords, np.ones(len(x_coords))))
    rotated_coords = np.dot(rotation_matrix, homogeneous_coords)
    return rotated_coords[0, :], rotated_coords[1, :]


def plot_lidar_annotations(x_centers_car, y_centers_car, widths_car, lengths_car, rotations_z_car,
                           x_centers_node, y_centers_node, widths_node, lengths_node, rotations_z_node):
    global_max_range = 30
    num_circles = 6
    step_size = global_max_range / num_circles
    step_size = max(int(np.ceil(step_size)), 1)
    ticks = np.arange(-global_max_range, global_max_range + step_size, step_size)

    # Rotate car bounding boxes
    rotated_boxes_car = rotate_bounding_boxes(x_centers_car, y_centers_car, widths_car, lengths_car, rotations_z_car)
    # Create Car Dataset Plot
    fig_car, ax_car = plt.subplots(figsize=(8, 8), facecolor='black')
    ax_car.set_facecolor('black')
    add_radar_guidelines(ax_car, global_max_range)
    add_rotated_bounding_boxes(ax_car, rotated_boxes_car)
    ax_car.set_xlim(-global_max_range+10, global_max_range)
    ax_car.set_ylim(-global_max_range, global_max_range)
    ax_car.set_aspect('equal', adjustable='box')
    ax_car.set_title("Car - Local Map", fontsize=18, color='white', pad=20)
    ax_car.set_xticks(ticks)
    ax_car.set_yticks(ticks)
    plt.show()
    plt.close(fig_car)

    # Rotate car bounding boxes
    rotated_boxes_node = rotate_bounding_boxes(x_centers_node, y_centers_node, widths_node, lengths_node, rotations_z_node)
    # Create Car Dataset Plot
    fig_node, ax_node = plt.subplots(figsize=(8, 8), facecolor='black')
    ax_node.set_facecolor('black')
    add_radar_guidelines(ax_node, global_max_range)
    add_rotated_bounding_boxes(ax_node, rotated_boxes_node)
    ax_node.set_xlim(-global_max_range, global_max_range)
    ax_node.set_ylim(-global_max_range, global_max_range)
    ax_node.set_aspect('equal', adjustable='box')
    ax_node.set_title("Node - Local Map", fontsize=18, color='white', pad=20)
    ax_node.set_xticks(ticks)
    ax_node.set_yticks(ticks)
    plt.show()
    plt.close(fig_car)


def main():
    # Load the datasets from the CSV files
    data_car_path_person_1 = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_1_annotations_car.csv"
    data_car_path_person_2 = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_2_annotations_car.csv"

    data_node_path_person_1 = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_1_annotations_node.csv"
    data_node_path_person_2 = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_2_annotations_node.csv"
    data_node_path_person_3 = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_3_annotations_node.csv"

    # data_node_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-720_id-869_fixed_person_1_annotations_node.csv"

    data_car_person_1 = pd.read_csv(data_car_path_person_1)
    car_person_1 = data_car_person_1.iloc[0]

    data_car_person_2 = pd.read_csv(data_car_path_person_2)
    car_person_2 = data_car_person_2.iloc[0]


    data_node_person_1 = pd.read_csv(data_node_path_person_1)
    node_person_1 = data_node_person_1.iloc[0]

    data_node_person_2 = pd.read_csv(data_node_path_person_2)
    node_person_2 = data_node_person_2.iloc[0]

    data_node_person_3 = pd.read_csv(data_node_path_person_3)
    node_person_3 = data_node_person_3.iloc[0]

    # Rotation matrix for transforming LIDAR coordinates
    theta_rotation_lidar_to_local = np.deg2rad(-15)
    R_lidar_to_local = [[np.cos(theta_rotation_lidar_to_local), -np.sin(theta_rotation_lidar_to_local), 0],
                        [np.sin(theta_rotation_lidar_to_local), np.cos(theta_rotation_lidar_to_local), 0],
                        [0, 0, 1]]



    # Loop through all rows of the datasets
    # for i in tqdm(range(min(len(data_car), len(data_node))), desc="Processing rows"):
    for i in tqdm(range(0, 1), desc="Processing rows"):
        # Collect and organize car data
        x_centers_car = [car_person_1['X_Center'], car_person_2['X_Center'], 3.719, -6.7423, -6.3682, -8.6951, -4.0587, -2.2538]
        y_centers_car = [car_person_1['Y_Center'], car_person_2['Y_Center'], 3.5261, 4.8329, 2.3938, -5.1920, -6.9455, -11.4903]
        widths_car = [0.6, 0.6, 0.6, 1.8, 1.8, 1.8, 1.8, 1.8]
        lengths_car = [0.6, 0.6, 0.6, 4.9, 4.9, 4.9, 4.9, 4.9]
        rotations_z_car = [0, 0, 0, 10.45, 11.61, 18.24, 16.16, 19.37]

        # Collect and organize node data
        x_centers_node = [node_person_1['X_Center'], node_person_2['X_Center'], node_person_3['X_Center'], -5.4945, -14.6857]
        y_centers_node = [node_person_1['Y_Center'], node_person_2['Y_Center'], node_person_3['Y_Center'], 18.3789, 26.5513]
        widths_node = [0.6, 0.6, 0.6, 1.8, 1.8]
        lengths_node = [0.6, 0.6, 0.6, 4.9, 4.9]
        rotations_z_node = [0, 0, 0, 34, 43]


        x_centers_car_rotated, y_centers_car_rotated = rotate_all_coordinates(x_centers_car, y_centers_car,
                                                                              R_lidar_to_local)

        filename = f"./plot_row_{i}.png"

        # Plot the lidar annotations with rotated bounding boxes
        plot_lidar_annotations(x_centers_car_rotated, y_centers_car_rotated, widths_car, lengths_car, rotations_z_car,
                               x_centers_node, y_centers_node, widths_node, lengths_node, rotations_z_node)


# Run the script
if __name__ == "__main__":
    main()