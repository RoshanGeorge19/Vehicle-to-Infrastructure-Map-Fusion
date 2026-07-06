# lidar_processing

Ground-plane removal, known-static-object filtering, and DBSCAN clustering
used to detect objects (pedestrians, parked cars) in the raw LiDAR point
clouds before they are annotated/geolocated.

## `lidar_clustering.py` -- `--format las|pcd --directory <dir> [--output-csv <path>] [--visualize]`
Scenario 2 clustering pipeline (merged from two near-duplicate originals that
differed only in reading `.las` vs `.pcd` files -- see root README).

| Function | Purpose |
|---|---|
| `readPointCloud(directory, pcd_file, vis)` | Loads a point cloud file, colouring points by distance from the sensor if it's a `.pcd`. |
| `filterDistanceAndGround(point_cloud, distance_lim, ground_thres, vis)` | Drops points beyond `distance_lim` metres, then RANSAC-segments and removes the ground plane. |
| `clusterObjects(point_cloud, method, vis)` | DBSCAN-clusters the remaining points into candidate objects. |
| `getStaticObj(pcd_file, original_filtered_point_cloud, vis)` | Returns the known static-object oriented bounding boxes for this scenario's parking lot (parked cars, traffic signs) so they can be excluded from clustering. |
| `removeStaticObj(filtered_point_cloud, static_bounding_boxes, vis)` | Removes points that fall inside any static-object bounding box. |
| `getBbox(pcd_file, original_filtered_point_cloud, static_bounding_boxes, point_cloud, labels, unique_labels, vis)` | Builds an oriented bounding box for each DBSCAN cluster. |
| `drawBevPoint(bounding_boxes, vis)` | Plots a bird's-eye-view scatter of bounding box centers. |
| `main(file_format, directory, output_csv, vis)` | Runs the full pipeline over every frame in `directory` and writes bounding box centers/extents to `output_csv`. |

## `lidar_clustering_scenario_3.py` -- `--directory <dir> [--output-csv <path>]`
Scenario 3 variant with different static objects (a pole and a parked car
rather than Scenario 2's cars/signs) and a larger distance limit (100 m vs 25
m). Imports `filterDistanceAndGround`, `clusterObjects`, `removeStaticObj`,
`getBbox`, and `drawBevPoint` from `lidar_clustering.py` rather than
duplicating them; only `readPointCloud` and `getStaticObj` are redefined.

## `simple_ground_removal.py`
Standalone ground-plane removal utility for a directory of `.las` files
(distance filtering + RANSAC ground segmentation), writing the filtered
clouds back out as `.las`.

| Function | Purpose |
|---|---|
| `readPointCloud(directory, las_file, vis)` | Loads a `.las` file into an Open3D point cloud. |
| `filterDistanceAndGround(point_cloud, distance_lim, ground_thres, vis)` | Same distance + RANSAC ground filtering as above. |
| `savePointCloud(point_cloud, directory_save, file_name)` | Writes a point cloud back to `.las`. |
| `main()` | Batch-processes every file in a directory. |

## `makeVideo.py`
`create_video_from_images(image_folder, video_name, fps)` -- stitches a
folder of `.png` frames (e.g. clustering visualisation screenshots) into an
`.mp4` video.
