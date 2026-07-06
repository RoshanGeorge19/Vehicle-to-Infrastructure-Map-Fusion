import laspy
import matplotlib.pyplot as plt
import numpy as np
import csv
from wp2.geo_utils import GeoTransformer
from tqdm import tqdm

node_geolocate_object = GeoTransformer.node_geolocate_object

def plot_lidar_data(x, y, intensity):
    plt.scatter(x, y, c=intensity, s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Intensity')
    plt.grid(True)
    plt.show()

def main():
    las_file_path = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario2-1/car_cepton_overlap_1/las_out_id-259_id-308/1099.644-Cepton_Scene-2.las'

    geoTransformer = GeoTransformer()
    compass_heading = 245.1

    las = laspy.read(las_file_path)

    points = (las.x, las.y, las.z)
    x, y, z = points
    intensity = las.intensity

    plot_lidar_data(x, y, intensity)

    mask = (np.abs(x) <= 10) & (np.abs(y) <= 20)
    x_filtered, y_filtered, z_filtered = x[mask], y[mask], z[mask]
    filtered_intensity = intensity[mask]

    plot_lidar_data(x_filtered, y_filtered, filtered_intensity)

    GPS_base_curr = (53.29047403, -9.07095837)
    gps_points = []

    coords = np.vstack((x_filtered, y_filtered, z_filtered)).T
    for i in tqdm(range(len(coords))):
        gps_point = node_geolocate_object(GPS_base_curr, compass_heading, coords[i])
        gps_points.append([1, f"node_{i}", gps_point[0], gps_point[1], gps_point[2], 1])  # Colour_ID=1, Show=1

    output_file = "G:/Documents/Pycharm Projects/Work_Package_1/src/data_processing/gps_points.csv"

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Colour_ID", "Name", "Longitude", "Latitude", "Altitude", "Show"])  # Header
        writer.writerows(gps_points)  # Data content

if __name__ == "__main__":
    main()