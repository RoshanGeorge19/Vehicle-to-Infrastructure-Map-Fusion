import glob
import natsort
import os
from tqdm import tqdm

las_loc = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/pcd_out/"

files = glob.glob(las_loc + "*.pcd")
files = natsort.natsorted(files)

for index, file in tqdm(enumerate(files), total=len(files)):
    file_time = os.path.basename(file).split(".")[0]
    file_time = float(file_time)
    new_time = (file_time / 1000000) - 1653642000
    new_file = f"{new_time:.3f}-Cepton_Scene-3-2.pcd"

    # Generate the full path for the current file and the new file
    curr_file_path = os.path.join(las_loc, file)
    new_file_path = os.path.join(las_loc, new_file)

    # Rename the file
    os.rename(curr_file_path, new_file_path)

