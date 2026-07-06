"""Merges all .avi files in a directory (sorted by filename) into a single output video."""
import argparse
import glob
import os

import cv2
from tqdm import tqdm


def main(dir_path, output_file):
    avi_files = sorted(glob.glob(os.path.join(dir_path, '*.avi')))

    cap = cv2.VideoCapture(avi_files[0])
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

    for avi_file in tqdm(avi_files, desc="Merging files"):
        cap = cv2.VideoCapture(avi_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Processing frames", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True, help='Directory containing the .avi files to merge')
    parser.add_argument('--output-file', default='merged_output.avi', help='Path to write the merged video')
    args = parser.parse_args()
    main(args.dir, args.output_file)
