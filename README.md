# The Impact of Pose Alignment Errors on a Classical Late Infrastructure-Vehicle Collaboration Framework Using Experimental Data

Code accompanying the paper published in *IEEE Open Journal of Vehicular
Technology* (2025). This repository investigates how pose alignment errors
(V2X transmission delay, GPS heading error, and LiDAR sensor calibration
(rotation/translation) error) degrade a late-fusion vehicle-infrastructure
(V2I) collaborative perception system, using real experimental LiDAR/GPS
data rather than simulation.

> Accurate perception and localization of dynamic objects is crucial for
> autonomous systems to navigate complex environments and avoid collisions.
> However, onboard sensing struggles with complex environments due to the
> challenges associated with occlusion, hindering progress in achieving
> advanced levels of vehicle autonomy in these environments. Cooperative
> driving automation has emerged as an enabling technology for the
> development of safer vehicles by using shared perception information, thus
> enhancing the situational awareness of the ego-vehicle. Alignment of
> shared perceptual information is a crucial task within this cooperative
> framework, and any pose errors can compromise the safety of the system.
> This paper investigates the real-world impact of pose alignment errors on
> vehicle-infrastructure collaborative driving by identifying, isolating,
> and analyzing individual error sources within a late collaboration
> framework. Our proposed method creates a shared global V2X environmental
> model by fusing LiDAR object representations from individual agents. We
> examine the sensitivity of this global map to common real-world error
> sources, including V2X communication delays, GPS positioning uncertainty,
> and sensor calibration errors.

Two agents are considered: an ego-vehicle (**car**) and a **Fixed Sensor
Node (FSN)** which is a roadside infrastructure unit with an elevated,
occlusion-resistant view of the scene. Each agent independently detects
pedestrians with its own LiDAR, and object-level detections are fused into a
shared global (GPS/ECEF) map (late collaboration). The experiments
systematically inject each error source in isolation and measure the resulting Euclidean displacement between the car's and
FSN's detections of the same pedestrian, to quantify how much each error
degrades map fusion accuracy at different distances from the sensors.

## Citation

If you use this code, please cite the paper:

```bibtex
@ARTICLE{11087629,
  author={George, Roshan and Molloy, Dara and Brophy, Tim and O'Grady, William and Mullins, Darragh and Jones, Edward and Deegan, Brian and Glavin, Martin},
  journal={IEEE Open Journal of Vehicular Technology},
  title={The Impact of Pose Alignment Errors on a Classical Late Infrastructure-Vehicle Collaboration Framework Using Experimental Data},
  year={2025},
  volume={6},
  number={},
  pages={2101-2130},
  keywords={Collaboration;Calibration;Pedestrians;Delays;Safety;Location awareness;Laser radar;Accuracy;Synchronization;Robustness;V2X;V2I;cooperative intelligent transportation systems (C-ITS);map fusion;infrastructure sensing;roadside units;collaborative driving automation},
  doi={10.1109/OJVT.2025.3591210}}
```

If you use the G-MIND dataset, please also cite the dataset paper
([IEEE Xplore](https://ieeexplore.ieee.org/document/11316264)):

```bibtex
@ARTICLE{11316264,
  author={Molloy, Dara and George, Roshan and Brophy, Tim and Deegan, Brian and Mullins, Darragh and Ward, Enda and Horgan, Jonathan and Eising, Ciaran and Denny, Patrick and Jones, Edward and Glavin, Martin},
  journal={IEEE Open Journal of Vehicular Technology},
  title={G-MIND: Galway Multimodal Infrastructure Node Dataset for Intelligent Transportation Systems},
  year={2026},
  volume={7},
  number={},
  pages={491-509},
  keywords={Sensors;Cameras;Pedestrians;Reliability;Thermal sensors;Roads;Laser radar;Collaboration;Automobiles;Vehicle-to-everything;V2X;V2I;cooperative intelligent transportation systems (C-ITS);infrastructure sensing;roadside units;automated mobility;collaborative driving automation},
  doi={10.1109/OJVT.2025.3648251}}
```

## Repository structure

Each `src/` subfolder has its own README describing every script and
function in it.

| Folder | Stage | Description |
|---|---|---|
| [`src/wp2/`](src/wp2/README.md) | shared | Core `GeoTransformer`/`Plotter` classes used throughout -- GPS/ECEF/LiDAR-frame conversions and shared plotting helpers. |
| [`src/preprocessing/`](src/preprocessing/README.md) | 1 | Raw capture cleanup: timestamp reformatting, RTK GPS merging, annotation-to-GPS matching. |
| [`src/geolocalisation/`](src/geolocalisation/README.md) | 2 | Converts car/node detections and point clouds into the shared global GPS frame (Fig. 5 in the paper). |
| [`src/lidar_processing/`](src/lidar_processing/README.md) | 2 | Ground removal, static-object filtering, and DBSCAN clustering to detect objects in raw LiDAR frames. |
| [`src/camera_processing/`](src/camera_processing/README.md) | 2 | Fisheye camera undistortion/projection and video utilities. |
| [`src/matlab/`](src/matlab/README.md) | 2 | Ground-truth cuboid annotation and ICP-based point cloud alignment (MATLAB). |
| [`src/verification/`](src/verification/README.md) | 3 | Sanity checks of SLAM trajectory / GPS interpolation / annotations against ground truth. |
| [`src/experiments/`](src/experiments/README.md) | 4 | **The paper's core experiments** -- sweeps delay, heading, rotation, and translation errors and measures map fusion accuracy (Section IV, Figs. 14-34). |
| [`src/analysis/`](src/analysis/README.md) | 4 | Supporting distance/error metrics (Euclidean, Mahalanobis, timestamp jitter, bounding-box point density). |
| [`src/visualisation/`](src/visualisation/README.md) | 5 | Plots detections over satellite imagery and local bird's-eye-view maps. |

## Dataset

Experiments use the **Galway Multimodal Infrastructure Node Dataset
(G-MIND)**, collected in an open parking lot with the FSN (3 RGB cameras, an
event camera, a thermal camera, 2 LiDARs, a radar, and a weather station) and
an ego-vehicle equipped with a roof-mounted Velodyne VLP-16 LiDAR. Ground
truth pose was obtained via RTK GPS (EMLID Reach M2, centimetre-level
accuracy) fused with LiDAR SLAM (LOAM). Only the VLP-16 (car) and Cepton P60
(FSN) LiDAR data and their manual pedestrian annotations were used for this
study.

G-MIND is now published as its own dataset paper:
[G-MIND: Galway Multimodal Infrastructure Node Dataset for Intelligent Transportation Systems](https://ieeexplore.ieee.org/document/11316264)
(IEEE Open Journal of Vehicular Technology, 2026). The dataset itself can be
downloaded from
[IEEE DataPort](https://ieee-dataport.org/documents/galway-multimodal-infrastructure-node-dataset).

Raw and intermediate data (camera frames, LiDAR point clouds, GPS logs,
~80GB) are **not included** in this repository. Scripts expect the following
layout:

```
data/
  camera_data/
  lidar_data/CAR/...      # vehicle-mounted LiDAR
  lidar_data/Cepton/...   # roadside FSN LiDAR
  lidar_data/Velodyne/...
  dangan_surveying_12-06-24/emlid_csv/   # RTK GPS survey points
```

## Installation

```bash
git clone <this-repo-url>
cd Work_Package_1
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.9+. MATLAB scripts in `src/matlab/` additionally require
MATLAB with the Lidar Toolbox (`groundTruthLidar`, `pcregistericp`,
`segmentGroundSMRF`, `lasFileReader`) and Computer Vision Toolbox.

## Usage

Scripts are organized as Python packages under `src/`. Run any script as a
module from the `src/` directory so that its imports (e.g.
`from wp2.geo_utils import GeoTransformer`) resolve correctly:

```bash
cd src
python -m preprocessing.emlid_rtk_processing
python -m geolocalisation.car_node_geo_localisation
python -m experiments.scenario_3_exp1_delay
python -m experiments.scenario_3_results
```

Most scripts read/write data at hardcoded paths reflecting the original
capture layout (e.g. `G:/Documents/Pycharm Projects/Work_Package_1/data/...`).
Update the path constants near the top of each script to point at your
own copy of the data before running.

A handful of near-duplicate per-axis/per-scenario scripts from the original
research codebase were merged into single parameterized scripts:

```bash
python -m experiments.scenario_3_exp3_rotation --axis x|y|z
python -m experiments.scenario_3_exp4_translation --axis x|y|z
python -m geolocalisation.extend_corners_car_parking_spots --corner top-left|bottom-right
python -m lidar_processing.lidar_clustering --format las|pcd --directory <path> [--visualize]
```

## License

The paper is published open-access under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). See the paper for full
terms.
