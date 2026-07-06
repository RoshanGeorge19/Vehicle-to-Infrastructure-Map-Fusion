# experiments

The core sensitivity experiments from the paper (Section IV): each script
sweeps one synthetic map-error source (transmission delay, GPS heading error,
LiDAR calibration rotation/translation) over a range of magnitudes, geolocates
the car and node detections at each magnitude, and records the Euclidean
distance between the two detections (the map-fusion accuracy metric,
Eq. 6). Each experiment pickles its results; the matching `_results.py`
script (where present) reloads the pickle and produces the paper's plots.

| Script | Paper section / figure | Purpose |
|---|---|---|
| `scenario_2_exp1_delay.py` | Sec. IV-B baseline (Scenario 2 site) | Sweeps transmission delay between car and node detections in the Scenario 2 dataset. Includes a superseded `geo_localize_car_objects_old` kept alongside the current implementation for reference. |
| `scenario_3_exp1_delay.py` / `scenario_3_exp1_delay_results.py` | Fig. 18 | Sweeps transmission delay (100-1000 ms) in Scenario 3 and plots Euclidean distance vs. delay, binned by distance-to-object range. |
| `scenario_3_exp2_heading.py` / `scenario_3_exp2_heading_results.py` | Fig. 19, 20, 31 | Sweeps synthetic GPS compass heading error (0-2 deg) applied to the node's geolocation. |
| `scenario_3_exp3_rotation.py --axis x\|y\|z` | Fig. 21, 22, 23, 32 | Sweeps a synthetic calibration rotation error (0-5 deg) about the given axis, applied to the node LiDAR-to-GPS transform. Merged from three per-axis originals -- see root README. |
| `scenario_3_exp4_translation.py --axis x\|y\|z` | Fig. 24, 25, 26, 27, 33, 34 | Sweeps a synthetic calibration translation error (0-5 m) along the given axis. Merged from three per-axis originals. |
| `scenario_3_results.py` | Fig. 14, 16, 18, 19, 21, 22, 24, 26, 28 and the global-map overlay figures (23, 25, 27, 30-34) | The main results/plotting script for the paper: loads every experiment's pickled output and the manual annotations, and reproduces essentially all of Section IV's figures (baseline performance, per-error-source plots, annotated global-map overlays, bounding-box point-count-vs-distance, and the Fig. 28 latency-to-positional-offset simulation). See the function-level breakdown below. |
| `viva_slides_figure.py` | -- | Presentation-figure variant of the geolocation pipeline (ECEF/ENU conversion helpers plus a parameter-sweep GPS export) produced for a conference slide deck; not part of the paper's main results but kept for reference. |

## `scenario_3_results.py` function reference

| Function | Produces |
|---|---|
| `plot_baseline(delay_pkl, vehicle_annotations, node_annotations)` | Fig. 14 -- baseline (zero-error) map fusion accuracy vs. distance. |
| `plot_delay(delay_pkl, ...)` | Fig. 18 -- Euclidean distance vs. transmission delay. |
| `plot_heading(heading_pkl)` | Fig. 19 -- Euclidean distance vs. GPS compass heading error. |
| `plot_x_rot` / `plot_y_rot` / `plot_z_rot` | Fig. 21 / -- / 22 -- Euclidean distance vs. rotation error about each axis. |
| `plot_x_tran` / `plot_y_tran` / `plot_z_tran` | Fig. 24 / 26 / -- -- Euclidean distance vs. translation error along each axis. |
| `plot_global_map`, `plot_delay_global_map[_old]`, `plot_heading_rotation_global_map[_2]`, `plot_point_global` | The annotated aerial/global-map figures (Figs. 23, 25, 27, 29-34) showing detections plotted over the site's satellite image. |
| `node_annotation_quality()` / `car_annotation_quality()` | Fig. 16 -- number of LiDAR points captured per detection vs. distance from sensor (bounding-box uncertainty analysis, Sec. III-G). |
| `plot_bbox_uncertainty()` | Bounding-box uncertainty discussion figure. |
| `plot_sim_delay_offset(log_scale=False)` | Fig. 28 -- simulated positional offset of an object for a given transmission latency, at several object speeds. |
| `rotate_all_coordinates(...)`, `plot_local_map(...)`, `plot_csv_data(...)` | Shared helpers for rotating/plotting the local (non-GPS) bird's-eye-view maps. |

Run any script as a module from `src/`, e.g.:
```
python -m experiments.scenario_3_exp3_rotation --axis z
python -m experiments.scenario_3_exp2_heading_results   # loads the pickle and plots it, where a _results.py counterpart exists
```
Note: only the delay and heading experiments have a matching `_results.py`
script in the original codebase; the rotation/translation sweeps are instead
plotted centrally by `scenario_3_results.py`.
