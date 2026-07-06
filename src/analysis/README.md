# analysis

Distance and error-metric utilities used to quantify agreement between car
and node detections, and to sanity-check the raw GPS/annotation data feeding
into the experiments.

| Script | Purpose |
|---|---|
| `get_distances_detected_car.py` | Reads a car/node geolocation results CSV and computes the ECEF Euclidean distance (via `nvector`) between each pair of matched car/node object detections -- the same metric as Eq. 6 in the paper, run standalone against a results file. |
| `merge_car_node_detections.py` | Time-aligns (`merge_asof`) a car object annotation CSV with the corresponding node annotation CSV, dropping redundant rotation/trajectory columns, to produce a single merged detection table. |
| `mahalanobis_distance.py` | `mahalanobis_distance(p, mean, inv_cov_matrix)` computes the Mahalanobis distance of a node detection from the distribution of car detections (mean + covariance of car GPS positions) -- an alternative, covariance-aware distance metric considered but not used as the paper's primary metric (see Sec. III-H for why Euclidean distance was chosen instead). |
| `time_error_analysis.py` | Checks GPS logger timestamp consistency: computes each row's expected timestamp (assuming a fixed 0.5 s sample interval) vs. its actual timestamp, and plots the distribution of timing jitter per contiguous capture group. |
| `get_azimuth_from_gps.py` | Computes the bearing/azimuth between consecutive GPS trajectory points (`geolocate_object` mirrors `wp2.geo_utils.GeoTransformer.node_geolocate_object`, kept here as a standalone script for azimuth-focused analysis). |
| `find_points_in_cuboid.py` | For each annotated bounding cuboid, counts how many raw LiDAR points from the corresponding `.las` frame fall inside it, and plots point count vs. distance from the sensor -- the bounding-box/point-density uncertainty analysis referenced in the paper's Sec. III-G (Fig. 12, Fig. 16). Functions: `read_annotations(csv_file)`, `load_las_file(las_path)`, `is_valid_cuboid(row)`, `count_points_in_cuboid(x, y, z, row)`, `calculate_distance(row)`, `process_annotations(directory, annotations)`, `plot_points_vs_distance(ax, filtered_df, title)`, `process_and_plot(ax, directory, csv_file, title)`. |
