# camera_processing

Fisheye camera undistortion/projection and video utilities for the
car-mounted RGB camera (Fig. 4a in the paper).

## `projection.py`
Camera projection models, ported from the Woodscape fisheye camera toolkit.
Not specific to this project's data -- a general-purpose projection library.

| Item | Purpose |
|---|---|
| `ensure_point_list(points, dim, concatenate=True, crop=True)` | Validates/reshapes an array of N-D points into a consistent list format. |
| `class Projection` | Abstract base defining `project_3d_to_2d` / `project_2d_to_3d`. |
| `class CylindricalProjection(Projection)` | Cylindrical lens model (used as the *output* projection when undistorting fisheye video, so the horizon appears straight). |
| `class RadialPolyCamProjection(Projection)` | 4th-order radial polynomial fisheye lens model (the *input* model matching the physical camera, calibrated in `front_camera_config.json`). |
| `class Camera` | Combines a lens `Projection` with extrinsic rotation/translation and image size/principal point, and projects world points to/from image pixels. |
| `create_img_projection_maps(source_cam, destination_cam)` | Builds the `cv2.remap` pixel-mapping arrays to warp an image captured with `source_cam`'s lens model into `destination_cam`'s. |
| `read_cam_from_json(path)` | Loads a `Camera` (intrinsics + extrinsics) from a calibration JSON file such as `front_camera_config.json`. |

## `front_camera_config.json`
Calibration for the vehicle's front-mounted fisheye camera: radial
polynomial distortion coefficients, resolution (1280x966), and the
extrinsic rotation/translation of the camera relative to the vehicle.

## `avi_car_fisheye_undistort.py` -- `--avi-file <path> [--cam-config front_camera_config.json] [--output-file undistorted_output.avi]`
Reads a fisheye `.avi` recording and `front_camera_config.json`, builds a
cylindrical target camera with a centered horizon (`make_cylindrical_cam`),
and writes an undistorted version of the video.

## `avi_car_merge.py` -- `--dir <path> [--output-file merged_output.avi]`
Concatenates every `.avi` file in a directory (sorted by filename) into a
single output video -- used to stitch together the car's per-segment camera
recordings.

## `video_to_frames.py` -- `--avi-file <path> [--output-dir png_images]`
Extracts every frame of a video to individually-numbered `.png` images.
