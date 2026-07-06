# verification

Sanity checks that validate intermediate outputs (SLAM trajectory, GPS
interpolation, pedestrian/static-object world coordinates) against manually
surveyed or annotated ground truth, before they're trusted as input to the
`experiments/`.

| Script | Purpose |
|---|---|
| `verify_vslam_trajectory.py` | Merges the SLAM odometry with object annotations on time, reconstructs each annotated object's world-frame coordinates by adding its sensor-relative position to the vehicle's odometry at that timestamp, and plots the result. Functions: `plot_trajectory_car(df)`, `verify_object_relative_to_lidar_car(df_annotations, df_odo)`, `verify_object_relative_to_lidar_node(df_annotations)`. |
| `verify_pedestrian_trajectory_car.py` | Merges a static-sign annotation with the car's SLAM trajectory (nearest-time match) and checks the reconstructed sign position against its known static location, as a check on SLAM drift. |
| `verify_traj_int_gps.py` | Computes ECEF Euclidean and great-circle distances between a set of manually-surveyed ground-truth points and their corresponding SLAM-interpolated-then-GPS-converted trajectory points, to validate the interpolation accuracy. |
