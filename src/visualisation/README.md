# visualisation

Plots detected/interpolated GPS points over satellite (GeoTIFF) imagery of
the test site, and renders local (non-GPS) bird's-eye-view maps of annotated
LiDAR detections.

| Script | Purpose |
|---|---|
| `gps_vis_geo_tiff.py` | The final, parameterized GeoTIFF visualisation script (kept from a sequence of iterative originals -- see root README): sweeps rotation/translation-X/translation-Y CSV result files and plots each parameter value's detections over the site's satellite image. Functions: `plot_csv_on_geotiff(...)`, `load_and_filter(csv_path)`, `handle_one(csv_path, title, out_png_name, rgb_image, extent, transformer)`. |
| `gps_vis_simple_own_map.py` | Interactive Dash app: loads a resampled GeoTIFF as the map background (rather than a Mapbox tile server) and plots GPS points on it as they're read from a CSV. Functions: `load_geotiff_rgb(path)`, `resample_geotiff(src_image, scale_factor)`, `update_output_div(...)` (Dash callback). |
| `make_local_map.py` | Renders a local (metres, not GPS) bird's-eye-view plot of car and node pedestrian detections with radar-style range guidelines, for visually comparing detections without needing satellite imagery. Functions: `add_radar_guidelines(ax, max_range, num_circles=6)`, `rotate_bounding_boxes(...)`, `add_rotated_bounding_boxes(ax, rotated_boxes)`, `rotate_all_coordinates(...)`, `plot_lidar_annotations(...)`. |
| `make_pose_alignment_figure.py` | Plots raw LiDAR point clouds (car and node) alongside their annotated object positions, used to visually check pose alignment between the two agents' point clouds. |

**Note:** `gps_vis_geo_tiff.py` and `gps_vis_simple_own_map.py` reference the
satellite GeoTIFF via a hardcoded personal OneDrive path
(`geotiff_path = r"C:/Users/.../image_modified.tif"`) -- point this at your
own copy of the site imagery before running.
