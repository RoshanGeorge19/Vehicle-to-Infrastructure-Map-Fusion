# geolocalisation

Converts car- and node-detected objects (and their LiDAR point clouds) from
each sensor's local frame into the shared global GPS/ECEF frame, and computes
the parking-spot survey geometry used as ground truth. This is the "map
fusion" step described in the paper's Section III (Vehicle/FSN Object
Geo-Localization, Fig. 5b/5c).

| Script | Purpose |
|---|---|
| `car_node_geo_localisation.py` | Core geolocation pipeline: loads the car trajectory/annotations and node annotations, geolocates both into global GPS coordinates, and computes the Euclidean distance between matched car/node object detections (the paper's baseline map-fusion accuracy metric, Fig. 14). Functions: `load_data()`, `preprocess_vehicle_data(...)`, `geo_localize_car_objects(...)`, `geo_localize_node_objects(...)`, `calculate_euclidean_distances(...)`. |
| `car_point_cloud_in_global.py` | Transforms a car LiDAR point cloud (LAS) into global GPS coordinates using the SLAM-derived local-to-global rotation, for visualisation/export. |
| `node_point_cloud_in_global.py` | Equivalent conversion for the roadside node's point cloud, using the node's GPS position and compass heading (`GeoTransformer.node_geolocate_object`) rather than a rotation-matrix calibration triplet. |
| `vslam_trajectory_coordinate_system_convert.py` | Converts the ego-vehicle's SLAM (LOAM) odometry through all three coordinate systems described in the paper's Fig. 5(a): LiDAR sensor frame -> local vehicle frame (fixed rotation) -> global GPS/ECEF frame (via calibration points). |
| `get_lidar_to_ecef_R_for_each_xy_pair.py` | Pre-computes the local-to-global rotation matrix `R` for every surveyed origin/X/Y calibration point triplet, and writes them to a CSV for later lookup (used by `wp2.geo_utils.GeoTransformer.getLidarToLocalCS_Rotation`). |
| `extend_corners_car_parking_spots.py --corner top-left\|bottom-right` | Extends the surveyed parking-spot corner grid outward by the mean spot spacing, to estimate GPS locations of corners that weren't directly surveyed. (Merged from two near-duplicate originals -- see root README.) |
| `get_center_gps_from_measured_corners.py` | Computes the center GPS point of each parking spot from four manually-surveyed corner points (`get_center_point(names)`). |
| `get_center_gps_from_interpolated_corners.py` | Same idea, but computes centers from the *extended/interpolated* corner points produced by `extend_corners_car_parking_spots.py` rather than direct survey measurements. |
| `estimate_sce3_car_node_gps_location.py` | One-off sanity check: given the MATLAB ICP registration translation vectors between the edge map, car VLP, and node VLP point clouds (see `src/matlab/edge_map_align_car_vlp.m` and `car_vlp_align_node_vlp.m`), estimates the car's/node's GPS location and compares it to survey ground truth. |

## Coordinate frames

All scripts here follow the paper's convention (Fig. 3, Fig. 7): a sensor's
raw point cloud is first rotated by a fixed angle (15 deg for the LiDARs in
this dataset) from the sensor frame into a **local** vehicle/node frame, then
transformed into the **global** ECEF/WGS84 frame using three known GPS
calibration points (an origin, a point along +Y, and a point along +X) via
`wp2.geo_utils.GeoTransformer.get_rotation`.
