"""
Scenario 2 LiDAR clustering pipeline: distance/ground filtering, static-object
(parked cars, traffic signs) removal via known bounding boxes, then DBSCAN
clustering of what remains.

Merged from lidar_clustering_scenario_2.py and lidar_clustering_scenario_2_v2.py,
which were identical apart from reading .las vs .pcd input and the visualisation
flag defaults. Run with `--format las|pcd` and optionally `--visualize`.

Scenario 3 has a separate script (lidar_clustering_scenario_3.py) since it adds
IOU-based bounding-box filtering not present here.
"""
import argparse
import os

import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def readPointCloud(directory, pcd_file, vis):
    print(pcd_file)
    file_path = os.path.join(directory, pcd_file)
    point_cloud = o3d.io.read_point_cloud(file_path)

    if pcd_file.endswith('.pcd'):
        distances = np.linalg.norm(np.asarray(point_cloud.points) - np.array([0, 0, 0]), axis=1)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        distances_normalized = (distances - min_distance) / (max_distance - min_distance)
        colors = plt.cm.inferno(distances_normalized)[:, :3]
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    if vis:
        o3d.visualization.draw_geometries([point_cloud])
    return point_cloud


def filterDistanceAndGround(point_cloud, distance_lim, ground_thres, vis):
    camera_position = np.array([0, 0, 0])
    distances = np.linalg.norm(np.asarray(point_cloud.points) - camera_position, axis=1)

    filtered_indices = np.where(distances <= distance_lim)[0]
    filtered_point_cloud = point_cloud.select_by_index(filtered_indices)

    plane_model, inliers = filtered_point_cloud.segment_plane(distance_threshold=ground_thres, ransac_n=3,
                                                              num_iterations=1000)

    outlier_cloud = filtered_point_cloud.select_by_index(inliers, invert=True)
    ground_plane = filtered_point_cloud.select_by_index(inliers, invert=False)
    if vis:
        o3d.visualization.draw_geometries([filtered_point_cloud])

    return filtered_point_cloud, outlier_cloud, ground_plane


def clusterObjects(point_cloud, method, vis):
    filtered_points = np.asarray(point_cloud.points)
    clusters = []

    if method == 'dbscan':
        epsilon = 1  # Distance threshold for DBSCAN
        min_samples = 25  # Minimum number of points in a cluster
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(filtered_points)

        unique_labels = set(labels)
        colors_labels = np.random.rand(len(unique_labels), 3)

        for label, color in zip(unique_labels, colors_labels):
            if label != -1:  # Exclude noise points
                cluster_indices = np.where(labels == label)[0]
                cluster_points = filtered_points[cluster_indices]

                cluster_cloud = o3d.geometry.PointCloud()
                cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
                cluster_cloud.paint_uniform_color(color)
                clusters.append(cluster_cloud)

        if vis:
            o3d.visualization.draw_geometries(clusters)
    else:
        print('Method Not Supported')
        return None

    return labels, unique_labels, colors_labels


def getStaticObj(pcd_file, original_filtered_point_cloud, vis):
    """Known static objects (parked cars, traffic signs) in the Scenario 2 parking lot."""
    static_bounding_boxes = []
    theta = 20
    rotation_x = np.array([[1, 0, 0], [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                           [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])

    def make_bbox(center, extent, angle_z):
        rotation_z = np.array([[np.cos(np.radians(angle_z)), -np.sin(np.radians(angle_z)), 0],
                               [np.sin(np.radians(angle_z)), np.cos(np.radians(angle_z)), 0],
                               [0, 0, 1]])
        rotation = np.dot(rotation_x, rotation_z)
        bbox = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        bbox.color = [255, 0, 0]
        return bbox

    static_bounding_boxes.append(make_bbox([-5.7, 18.4, 3.2], [4.4, 2.2, 1.5], 45))    # Parked_Car_1
    static_bounding_boxes.append(make_bbox([-7.6, 19.9, 3.8], [3.6, 2.0, 1.4], 45))    # Parked_Car_2
    static_bounding_boxes.append(make_bbox([4, 7.65, 0.1], [0.7, 0.75, 2.9], 0))       # Traffic_Sign_1
    static_bounding_boxes.append(make_bbox([0.95, 15.7, 2.4], [0.2, 0.24, 2.3], 0))    # Traffic_Sign_2

    if vis:
        vis_win = o3d.visualization.Visualizer()
        vis_win.create_window()
        vis_win.add_geometry(original_filtered_point_cloud)
        for bbox in static_bounding_boxes:
            vis_win.add_geometry(bbox)
        vis_win.get_render_option().point_size = 4
        vis_win.run()
        vis_win.capture_screen_image(f'{pcd_file[:-4]}.png')
        vis_win.destroy_window()

    return static_bounding_boxes


def removeStaticObj(filtered_point_cloud, static_bounding_boxes, vis):
    filtered_points = np.asarray(filtered_point_cloud.points)
    points_to_remove_indices = []
    for static_bbox in static_bounding_boxes:
        idx_in_bbox = static_bbox.get_point_indices_within_bounding_box(filtered_point_cloud.points)
        static_object_points = filtered_points[idx_in_bbox]
        static_object_points_reshaped = static_object_points.reshape(1, -1, 3)
        comparison = np.all(filtered_points[:, np.newaxis, :] == static_object_points_reshaped, axis=2)
        points_to_remove_indices.extend(np.where(comparison.any(axis=1))[0])

    static_removed_points = np.delete(filtered_points, points_to_remove_indices, axis=0)
    static_removed_points_cloud = o3d.geometry.PointCloud()
    static_removed_points_cloud.points = o3d.utility.Vector3dVector(static_removed_points)

    if vis:
        o3d.visualization.draw_geometries([static_removed_points_cloud], window_name="Static Removed Cloud")

    return static_removed_points_cloud


def getBbox(pcd_file, original_filtered_point_cloud, static_bounding_boxes, point_cloud, labels, unique_labels, vis):
    bounding_boxes = [static_bbox for static_bbox in static_bounding_boxes]

    filtered_points = np.asarray(point_cloud.points)
    for label in unique_labels:
        if label != -1:
            cluster_indices = np.where(labels == label)[0]
            cluster_points = filtered_points[cluster_indices]
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)
            center = (min_bound + max_bound) / 2
            extent = max_bound - min_bound

            theta = 20
            alpha = 0
            rotation_x = np.array([[1, 0, 0], [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                   [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
            rotation_z = np.array([[np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0],
                                   [np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0], [0, 0, 1]])
            rotation = np.dot(rotation_x, rotation_z)

            min_z = np.min(cluster_points[:, 2])
            center_adjusted = np.array([center[0], center[1], min_z + extent[2] / 2])
            extent_adjusted = np.array([extent[0], extent[1], extent[2] + center[2] - min_z])

            bbox = o3d.geometry.OrientedBoundingBox(center_adjusted, rotation, extent_adjusted)
            bbox.color = [0, 0, 0]
            bounding_boxes.append(bbox)

    if vis:
        vis_win = o3d.visualization.Visualizer()
        vis_win.create_window()
        vis_win.add_geometry(original_filtered_point_cloud)
        for bbox in bounding_boxes:
            vis_win.add_geometry(bbox)
        vis_win.get_render_option().point_size = 4
        vis_win.run()
        vis_win.capture_screen_image(f'{pcd_file[:-4]}.png')
        vis_win.destroy_window()

    return bounding_boxes


def drawBevPoint(bounding_boxes, vis):
    centers = [bbox.center[:2] for bbox in bounding_boxes]
    unique_centers = np.unique(centers, axis=0)
    num_colors = len(unique_centers)
    colormap = plt.cm.get_cmap('tab10', num_colors)

    for i, center in enumerate(unique_centers):
        color = colormap(i)
        matching_bboxes = [bbox for bbox in bounding_boxes if np.allclose(bbox.center[:2], center)]
        for bbox in matching_bboxes:
            plt.plot(center[0], center[1], marker='o', color=color)
            plt.text(center[0], center[1], f'({center[0]:.1f}, {center[1]:.1f})', fontsize=7,
                     verticalalignment='bottom', horizontalalignment='right', color='red')

    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim(-15, 15)
    plt.ylim(0, 25)
    if vis:
        plt.show()
        plt.close()


def main(file_format, directory, output_csv, vis):
    pcd_files = [f for f in os.listdir(directory) if f.endswith(f'.{file_format}')]

    df = pd.DataFrame(
        columns=['timestamp', 'center_x', 'center_y', 'center_z', 'extent_length', 'extent_width', 'extent_height',
                 'rotation_x', 'rotation_y', 'rotation_z'])

    for pcd_file in pcd_files:
        point_cloud = readPointCloud(directory, pcd_file, vis=False)
        dist_filt_point_cloud, filtered_point_cloud, ground_plane = filterDistanceAndGround(
            point_cloud, distance_lim=25, ground_thres=0.20, vis=vis)

        static_bounding_boxes = getStaticObj(pcd_file, dist_filt_point_cloud, vis=vis)
        static_removed_points_cloud = removeStaticObj(filtered_point_cloud, static_bounding_boxes, vis=vis)

        labels, unique_labels, _ = clusterObjects(static_removed_points_cloud, 'dbscan', vis=vis)
        getBbox(pcd_file, dist_filt_point_cloud, static_bounding_boxes, static_removed_points_cloud, labels, unique_labels, vis=vis)

        timestamp = pcd_file.split('-')[0]
        for bbox in static_bounding_boxes:
            center = bbox.center
            extent = bbox.extent
            new_row = pd.DataFrame({
                'timestamp': [timestamp],
                'center_x': [center[0]], 'center_y': [center[1]], 'center_z': [center[2]],
                'extent_length': [extent[0]], 'extent_width': [extent[1]], 'extent_height': [extent[2]],
                'rotation_x': [0], 'rotation_y': [0], 'rotation_z': [0]
            })
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['las', 'pcd'], required=True, help='Input point cloud file format')
    parser.add_argument('--directory', required=True, help='Directory containing the point cloud frames')
    parser.add_argument('--output-csv', default='static_bounding_boxes.csv', help='Where to write the bounding box CSV')
    parser.add_argument('--visualize', action='store_true', help='Show Open3D/matplotlib visualisations while processing')
    args = parser.parse_args()
    main(args.format, args.directory, args.output_csv, args.visualize)
