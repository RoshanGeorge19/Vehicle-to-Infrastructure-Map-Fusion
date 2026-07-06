# Generate a CSV file with the original file name and the index file name.
import os
import csv

# Define the directory
folder = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/node_velo_pcd_out/"


# Get a list of all .ply files in the directory
files = [f for f in os.listdir(folder) if f.endswith('.pcd')]

# Sort the files
files.sort()

print(files)

with open('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario3-2/point_cloud_data/car_las_file_index.csv', 'w', newline='') as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['Original_File_Name', 'Index_File_Name'])

    # Iterate over the files and write to the CSV
    for i, filename in enumerate(files, start=1):
        writer.writerow([filename, f'{str(i).zfill(6)}.pcd'])