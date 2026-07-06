import argparse

import cv2
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from camera_processing.projection import Camera, RadialPolyCamProjection, CylindricalProjection, read_cam_from_json, create_img_projection_maps


def make_cylindrical_cam(cam: Camera):
    """generates a cylindrical camera with a centered horizon"""
    assert isinstance(cam.lens, RadialPolyCamProjection)
    # creates a cylindrical projection
    lens = CylindricalProjection(cam.lens.coefficients[0])
    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler('zxz')
    # adjust all angles to multiples of 90 degree
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)
    # center horizon
    rot_zxz[1] = np.pi / 2
    # noinspection PyArgumentList
    return Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq='zxz').as_matrix(),
        translation=cam.translation,
        lens=lens,
        size=cam.size, principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio
    )


def main(avi_file, cam_config_file, output_file):
    fisheye_cam = read_cam_from_json(cam_config_file)
    cylindrical_cam = make_cylindrical_cam(fisheye_cam)

    cap = cv2.VideoCapture(avi_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

    map1, map2 = create_img_projection_maps(fisheye_cam, cylindrical_cam)
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, fisheye_image = cap.read()
        if not ret:
            break
        undistorted_frame = cv2.remap(fisheye_image, map1, map2, cv2.INTER_CUBIC)
        out.write(undistorted_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--avi-file', required=True, help='Path to the fisheye .avi video to undistort')
    parser.add_argument('--cam-config', default='front_camera_config.json', help='Path to the camera calibration JSON')
    parser.add_argument('--output-file', default='undistorted_output.avi', help='Path to write the undistorted video')
    args = parser.parse_args()
    main(args.avi_file, args.cam_config, args.output_file)
