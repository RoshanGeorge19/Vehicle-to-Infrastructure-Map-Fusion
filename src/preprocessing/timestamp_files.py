import glob
import os
import sys
from tqdm import tqdm

# folder = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/pcd_out/"
# files = glob.glob(folder + "*.pcd")
#
# for file in files:
#     # Extract the timestamp from the filename
#     timestamp_sec = float(os.path.splitext(os.path.basename(file))[0].split('-')[0])
#
#     # Add 529.1 to the timestamp
#     new_timestamp_sec = timestamp_sec + 529.1
#
#     # Generate the new filename
#     new_filename = f"{new_timestamp_sec:.3f}-Car_Scene-2.ply"
#
#     # Generate the full path for the new file
#     new_file_path = os.path.join(os.path.dirname(file), new_filename)
#
#     # Rename the file
#     os.rename(file, new_file_path)


folder = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/node_velo_pcd_out/"
files = glob.glob(folder + "*.pcd")

# start_epoch = 1653645547.54
start_epoch = 1653646079.67
day_epoch = 1653642000
# curr_epoch = (start_epoch - day_epoch)
curr_epoch = (start_epoch - day_epoch)

acq_rate = 10

for i in tqdm(range(len(files))):
    # VLP16, Thermal, CCTV.
    # source = f"{folder}{str(i + 1)}.png" # str input = 1.png, 10.png, 100.png, 1000.png, etc.
    source = f"{folder}{str(i + 1).zfill(5)}.pcd"  # str input = 001.png, 010.png, 100.png, 1000.png, etc.
    # dest = f"{folder}{round(curr_epoch, 3)}-CCTV-Scenario3-7.png"
    dest = dest = f"{folder}{curr_epoch:.3f}-Cepton_Velo_Scene-3-2.pcd"

    curr_epoch += (1 / acq_rate)

    # Cepton
    # source = files[i]
    # file_name = round(((float(os.path.basename(source)[:-4])/1000000)-1653609600), 3)
    # dest = f"{folder}{file_name:.3f}-Cepton-DarkArea-PersonCyclist.pcd"

    os.rename(source, dest)
