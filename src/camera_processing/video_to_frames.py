"""Extracts every frame of a video file to numbered .png images in an output directory."""
import argparse
import os

import cv2
from tqdm import tqdm


def main(avi_file, output_dir):
    cap = cv2.VideoCapture(avi_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f'frame_{i:06d}.png'), frame)

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--avi-file', required=True, help='Path to the video to split into frames')
    parser.add_argument('--output-dir', default='png_images', help='Directory to write the extracted frames to')
    args = parser.parse_args()
    main(args.avi_file, args.output_dir)
