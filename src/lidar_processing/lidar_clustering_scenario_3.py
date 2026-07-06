"""
Scenario 3 LiDAR clustering pipeline. Shares its distance/ground filtering,
DBSCAN clustering, static-object removal, and bounding box logic with
lidar_clustering.py (Scenario 2) -- only the static object definitions
(a pole and a parked car, vs. Scenario 2's parked cars and traffic signs) and
the processing distance limit differ, so those functions are imported instead
of duplicated here.
"""
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

from lidar_processing.lidar_clustering import (
    filterDistanceAndGround, clusterObjects, removeStaticObj, getBbox, drawBevPoint,
)


def readPointCloud(directory, pcd_file, vis):
    file_path = os.path.join(directory, pcd_file)
    point_cloud = o3d.io.read_point_cloud(file_path)
    if vis:
        o3d.visualization.draw_geometries([point_cloud])
    return point_cloud


def getStaticObj(pcd_file, original_filtered_point_cloud, vis):
    """Known static objects (a pole and a parked car) in the Scenario 3 site."""
    static_bounding_boxes = []
    theta = 20
    rotation_x = np.array([[1, 0, 0], [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                           [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])

    center_pole = [2.75343, 16.8522, 3.06198]
    extent_pole = [0.201084, 0.518986, 1.77923]
    rotation_pole = np.dot(rotation_x, np.eye(3))
    bbox_pole = o3d.geometry.OrientedBoundingBox(center_pole, rotation_pole, extent_pole)
    bbox_pole.color = [255, 0, 0]
    static_bounding_boxes.append(bbox_pole)

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


def main(directory, output_csv=None):
    pcd_files = [file for file in os.listdir(directory) if file.endswith('.pcd')]
    centers = []

    for pcd_file in tqdm(pcd_files, desc="Processing point cloud files"):
        point_cloud = readPointCloud(directory, pcd_file, vis=False)
        dist_filt_point_cloud, filtered_point_cloud, ground_plane = filterDistanceAndGround(
            point_cloud, distance_lim=100, ground_thres=0.20, vis=False)

        static_bounding_boxes = getStaticObj(pcd_file, dist_filt_point_cloud, vis=False)
        static_removed_points_cloud = removeStaticObj(filtered_point_cloud, static_bounding_boxes, vis=False)

        labels, unique_labels, _ = clusterObjects(static_removed_points_cloud, 'dbscan', vis=False)
        bounding_boxes = getBbox(pcd_file, dist_filt_point_cloud, static_bounding_boxes, static_removed_points_cloud, labels, unique_labels, vis=False)
        results = list(filter(lambda x: x not in static_bounding_boxes, bounding_boxes))

        for bbox in results:
            centers.append([pcd_file, bbox.center[0], bbox.center[1], bbox.center[2]])

    if output_csv:
        import csv
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "X", "Y", "Z"])
            writer.writerows(centers)

    return centers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, help='Directory containing the Scenario 3 point cloud frames')
    parser.add_argument('--output-csv', default=None, help='Optional path to write detected object centers to')
    args = parser.parse_args()
    main(args.directory, args.output_csv)
