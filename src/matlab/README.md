# matlab

MATLAB scripts for LiDAR point cloud annotation (ground-truth cuboid
labelling with the Lidar Labeler app), ICP-based alignment between the car,
node, and camera-derived edge-map point clouds, and exporting labels to CSV.
Requires MATLAB's Lidar Toolbox (`groundTruthLidar`, `lidarLabeler`,
`pcregistericp`, `segmentGroundSMRF`, `lasFileReader`) and Computer Vision
Toolbox.

| Script | Purpose |
|---|---|
| `car_vlp_align_node_vlp.m` | Aligns the car and node VLP-16 point clouds using ground-plane removal (SMRF) followed by ICP registration, outputting the translation vector between the two sensors' local frames (used as `T2` in `geolocalisation/estimate_sce3_car_node_gps_location.py`). |
| `edge_map_align_car_vlp.m` | Aligns the camera-derived edge map point cloud to the car's VLP-16 point cloud via ICP, given an initial translation guess (used as `T1` in `estimate_sce3_car_node_gps_location.py`). |
| `rotate_pcd_lidar_to_local_cs.m` | Applies the fixed -15 deg Z-axis rotation that converts the edge-map/car/node point clouds from their raw sensor frame into the shared local coordinate system (the MATLAB-side equivalent of `wp2.geo_utils.GeoTransformer.get_slam_rotated_trajectory`). |
| `rotate_point_cloud_check_bbox.m` | Verifies the above rotation on a single `.las` frame by comparing the point cloud before/after a +15 deg rotation. |
| `draw_cuboid.m` | Loads a `.las` frame and its ground-truth annotation `.mat` file, and renders the annotated 3D cuboid over the point cloud in top-down/side/front views. |
| `count_points_interactive.m` | Plots the number of LiDAR points inside each annotated cuboid across all frames, with click-to-inspect 3D visualisation of any point -- the interactive tool behind the paper's Fig. 12/16 bounding-box-uncertainty analysis. |
| `labels_to_csv_car.m` / `labels_to_csv_node.m` | Export a `groundTruthLidar` annotation object's per-frame object position/dimensions/rotation to a CSV (`session_converted_csv.csv`). The car and node variants differ in how the annotation is indexed (`{i,:}` cell vs. `(i,:)` table row) due to a difference in how each dataset's `groundTruthLidar` object stores its label data. |
| `fixed_cuboid_dimensions.m` | Overwrites a set of annotated cuboids' (variable) dimensions with a single fixed pedestrian bounding box size (0.6 x 0.5 x 1.7 m), to remove annotation-dimension noise from the analysis. |
| `reformat_annotations_testing_lidar_point_spikes.m` | Shifts an annotation's X/Z center by a fixed offset to compensate for a known point-cloud spike artefact at one capture site. |
| `add_cuboid_labels.m` | Copies a fixed object dimension (`static_sign_2`) from one ground-truth annotation file into another's label data. |
| `addTimestampsToLabelData.m` | Looks up and attaches the correct capture timestamp to each row of an extracted label CSV, using a separate frame-index-to-timestamp CSV and natural-sort matching (`natsort`, from the third-party `sort` toolbox on the MATLAB path). |

All scripts load data from hardcoded paths under `G:\Documents\Pycharm
Projects\Work_Package_1\data\...` -- update the `load(...)`/`pcread(...)`
paths at the top of each script before running.
