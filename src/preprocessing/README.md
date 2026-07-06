# preprocessing

One-off data preparation scripts that convert raw capture output (point cloud
dumps, RTK GPS logs, router GPS exports, annotation spreadsheets) into the
timestamp-aligned CSVs consumed by `geolocalisation/` and `experiments/`.
These are run once per capture session, in roughly the order listed below,
rather than being part of a repeatable pipeline.

| Script | Purpose |
|---|---|
| `timestamp_files.py` / `car_timestamp_reformat.py` / `cepton_timestamp_reformat.py` | Batch-rename captured point cloud files (`.las`/`.pcd`/`.ply`) from their raw sequential/epoch filenames to a common `<seconds_since_scenario_start>-<Scenario>.ext` naming scheme, so frames can be matched across the car and node LiDARs by timestamp. |
| `timestamp_index_gen.py` | Writes a CSV mapping each point cloud file's original filename to a sequential index filename, for tools that require sequentially-numbered input. |
| `router_GPS_time_to_unix_time_fixed.py` | Converts the GPS router's logged timestamps to Unix time in milliseconds (with a fixed one-hour offset correction) and re-saves the trajectory CSV. |
| `emlid_rtk_processing.py` | Merges the individual Emlid RTK survey CSVs (parking spot corners, road arrows, static signs, wooden posts) into one file, tagging each row with its source filename via `add_filename_column(df, filename)`. |
| `merge_trajectory_and_gps_on_time.py` | Merges the SLAM trajectory (X/Y/Z/roll/pitch/yaw) with the corresponding matched GPS fix, joined on rounded timestamp. |
| `add_bearing_and_climb.py` | Merges GPS bearing/climb/speed fields (computed elsewhere) onto the ground-truth GPS trajectory sheet, joined on time. |
| `match_annotations_with_gps.py` | For each annotated object detection, finds the nearest-in-time GPS fix and attaches it to the annotation row. |
| `get_delta_from_csl20.py` | One-off calibration calculation: derives the half-distance between two known survey points (`csl4`, `csl8`) and uses it, together with the local-to-global rotation from `wp2.geo_utils.GeoTransformer`, to estimate a GPS offset point. |

All scripts read/write CSVs at hardcoded paths that reflect the original
capture directory layout (`data/lidar_data/...`, `data/dangan_surveying_.../`)
-- update the paths at the top of each script to point at your own copy of
the raw data before running.
