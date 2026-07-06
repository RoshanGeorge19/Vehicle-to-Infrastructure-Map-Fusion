# wp2

Shared library used by nearly every other package in this repository. Contains
the coordinate-transformation math and the shared plotting helpers, so that
scenario/experiment scripts don't reimplement them.

## `geo_utils.py`

### `class GeoTransformer`
Converts between GPS (WGS84), ECEF, and sensor/LiDAR-local coordinate frames,
and computes the rotation matrix that aligns a sensor's local frame to the
global frame.

| Method | Description |
|---|---|
| `gps_to_ecef(lat, lon, alt)` | Converts a WGS84 GPS coordinate to ECEF (Earth-Centered, Earth-Fixed) XYZ. |
| `ecef_to_gps(x, y, z)` | Inverse of the above: ECEF XYZ back to (lon, lat, alt). |
| `normalize(vector)` *(static)* | Unit-normalizes a vector. |
| `lidar_to_ecef(lidar_point, ECEF0, R)` *(static)* | Projects a point from a sensor's local frame into ECEF, given the sensor's ECEF origin and rotation matrix `R`. |
| `get_rotation(ECEF0, ECEFX, ECEFY)` | Derives the 3x3 rotation matrix from local to global (ECEF) frame from three known GPS calibration points: an origin and points defining the local X and Y axes. |
| `get_slam_rotated_trajectory(df)` *(static)* | Applies the fixed -15 deg SLAM-frame-to-local-frame rotation (see `car_vlp_align_node_vlp.m`/`rotate_pcd_lidar_to_local_cs.m`) to a trajectory DataFrame with `X`, `Y`, `Z` columns. |
| `convert_slam_rotated_trajectory_to_gps(GPS0, GPSX, GPSY, GPS_start, df_rotated, df_car_base, df, geo_transformer)` *(static)* | Converts a locally-rotated SLAM trajectory into GPS coordinates using the calibration points and a known starting GPS fix. |
| `getLidarToLocalCS_Rotation(df_R_t_interval, time_n, geoTransformer, hard_values)` *(static)* | Looks up (or hardcodes, if `hard_values=True`) the local-to-global rotation matrix that applies at a given trajectory timestamp, from a table of per-interval calibration rotations. |
| `node_geolocate_object(lidar_origin_gps, compass_heading, object_lidar_coords)` *(static)* | Geolocates an object detected in the roadside node's LiDAR frame directly into GPS coordinates, using the node's GPS position and compass heading (bypasses ECEF, used where no local-frame calibration point triplet is available). |

### `class Plotter`
Matplotlib/Plotly plotting helpers shared by the experiment `_results.py`
scripts, so each one doesn't reimplement its own formatting.

| Method | Description |
|---|---|
| `format_tick_value(value, pos)` *(static)* | Matplotlib tick formatter callback. |
| `generic_plot(ax, x_data, y_data, title, y_label, formatter, delay_node, rotation=45, xlabel=None, color=None, label=None)` *(static)* | Generic line/scatter plot with consistent styling used across several result scripts. |
| `plot_euclidean_distances(eucl_dist_dict)` *(static)* | Plots car-vs-node Euclidean distance results across a swept error parameter (delay/heading/rotation/translation). |
| `plot_error_metrics(error_metrics_dict)` *(static)* | Plots summary error-metric statistics (mean/std/etc.) across the sweep. |
| `plot_relative_error(relative_error_dict)` *(static)* | Plots the relative error of each error magnitude against the zero-error baseline. |
| `plot_delay_diff_interactive(diff_df)` *(static)* | Interactive Plotly version of the delay-difference plot. |
