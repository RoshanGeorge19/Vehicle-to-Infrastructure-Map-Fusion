import pandas as pd
import glob
import natsort
import os
from tqdm import tqdm

# df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/fix_car_scene_2_names_with_timestamp.csv')
# las_loc = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/point_cloud_data/las_out/"
#
# files = glob.glob(las_loc + "*.las")
# files = natsort.natsorted(files)
#
# for file in files:
#     print(os.path.basename(file).split("/")[0])
#
# for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#     curr_file = row['Curr File Name']
#     new_file = f"{row['Time Short']:.3f}-Car_Scene-3.las"
#
#     # Generate the full path for the current file and the new file
#     curr_file_path = os.path.join(las_loc, curr_file)
#     new_file_path = os.path.join(las_loc, new_file)
#
#     # Rename the file
#     # os.rename(curr_file_path, new_file_path)

las_loc = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/node_velo_pcd_out_2/"
las_loc_2 = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/test/"
files = glob.glob(las_loc + "*.las")
files = natsort.natsorted(files)

# for file in files:
#     print(os.path.basename(file).split("/")[0])

for index, file in tqdm(enumerate(files), total=len(files)):
    # file_time = os.path.basename(file).split(".")[0]
    # file_time = float(file_time)
    # new_time = (file_time / 1000000) - 1653642000
    # new_time = file_time - 1653642000

    file_time = os.path.basename(file).split("-")[0]
    new_time = float(file_time) + 0.1
    new_file = f"{new_time:.3f}-Cepton_Velo_Scene-3-2.pcd"

    # Generate the full path for the current file and the new file
    curr_file_path = os.path.join(las_loc, file)
    new_file_path = os.path.join(las_loc_2, new_file)

    # Rename the file
    os.rename(curr_file_path, new_file_path)





