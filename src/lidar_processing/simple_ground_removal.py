import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import laspy  # For reading and writing LAS files (modern API)


def readPointCloud(directory, las_file, vis):
    # Construct the full file path
    file_path = os.path.join(directory, las_file)

    # Read the .las file using modern laspy
    with laspy.open(file_path) as las_file:
        las_data = las_file.read()
        # Access normalized point data (real-world x, y, z coordinates)
        points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Create an Open3D point cloud from the extracted points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud if needed
    if vis:
        o3d.visualization.draw_geometries([point_cloud])

    return point_cloud


def filterDistanceAndGround(point_cloud, distance_lim, ground_thres, vis):
    camera_position = np.array([0, 0, 0])
    distances = np.linalg.norm(np.asarray(point_cloud.points) - camera_position, axis=1)

    # Filter points that are within the specified distance limit
    filtered_indices = np.where(distances <= distance_lim)[0]
    filtered_point_cloud = point_cloud.select_by_index(filtered_indices)

    # Use RANSAC for ground plane segmentation
    plane_model, inliers = filtered_point_cloud.segment_plane(
        distance_threshold=ground_thres,
        ransac_n=3,
        num_iterations=1000
    )

    outlier_cloud = filtered_point_cloud.select_by_index(inliers, invert=True)  # Non-ground points
    ground_plane = filtered_point_cloud.select_by_index(inliers, invert=False)  # Ground points only

    # Visualize outlier cloud (non-ground points) if needed
    if vis:
        o3d.visualization.draw_geometries([outlier_cloud])

    return filtered_point_cloud, outlier_cloud, ground_plane


def savePointCloud(point_cloud, directory_save, file_name):
    # Ensure the save directory exists
    os.makedirs(directory_save, exist_ok=True)

    # Construct full file path for saving
    save_path = os.path.join(directory_save, file_name + ".las")

    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)

    # Create a header for the LAS file
    header = laspy.LasHeader(point_format=0, version="1.2")  # Point format 0: X, Y, Z only

    # Create a LasData object and assign coordinates
    las_data = laspy.LasData(header)
    las_data.x = points[:, 0]
    las_data.y = points[:, 1]
    las_data.z = points[:, 2]

    # Write to file
    las_data.write(save_path)

    print(f"Saved filtered point cloud to: {save_path}")


def main():
    directory = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/car_cepton_overlap_1/person_1/out/'
    directory = 'G:/Documents/Cepton SDK/python/tools/points/8759/'
    directory_save = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-3/car_cepton_overlap_1/'

    # Get all .las files in the input directory
    las_files = [file for file in os.listdir(directory) if file.endswith('.las')]

    # Process each file
    for las_file in tqdm(las_files, desc="Processing point cloud files"):
        point_cloud = readPointCloud(directory, las_file, vis=False)
        dist_filt_point_cloud, filtered_point_cloud, ground_plane = filterDistanceAndGround(point_cloud, distance_lim=500, ground_thres=0.2, vis=False)

        # Save filtered point cloud as a .las file
        savePointCloud(filtered_point_cloud, directory_save, os.path.splitext(las_file)[0])


if __name__ == "__main__":
    main()