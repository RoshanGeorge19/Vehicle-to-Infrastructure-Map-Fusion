import laspy
import matplotlib.pyplot as plt
import numpy as np
import csv
from wp2.geo_utils import GeoTransformer
from tqdm import tqdm

def plot_lidar_data(x, y, intensity):
    plt.scatter(x, y, c=intensity, s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Intensity')
    plt.grid(True)
    plt.show()

def main():
    las_file_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/car_cepton_overlap_1/las_out_id-307_id-356/1099.700-Car_Scene-2.las"
    geoTransformer = GeoTransformer()

    # Read LAS file using laspy 2.x
    las = laspy.read(las_file_path)

    x, y, z = las.x, las.y, las.z
    intensity = las.intensity

    plot_lidar_data(x, y, intensity)

    # Apply filtering
    mask = (np.abs(x) <= 20) & (np.abs(y) <= 20)
    x_filtered, y_filtered, z_filtered = x[mask], y[mask], z[mask]
    filtered_intensity = intensity[mask]

    plot_lidar_data(x_filtered, y_filtered, filtered_intensity)

    # ✅ Transformation matrix
    theta = np.deg2rad(-15)
    R_LidarToLocal = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]

    coords = np.vstack((x_filtered, y_filtered, z_filtered))
    points_local_cs = np.dot(R_LidarToLocal, coords).T

    plot_lidar_data(points_local_cs[:, 0], points_local_cs[:, 1], filtered_intensity)

    # ✅ Define GPS Reference Points
    GPS0 = (53.28989834, -9.07136142, 67.566)  # Origin
    GPSY = (53.28991859, -9.07135091, 67.552)  # Positive Y direction
    GPSX = (53.28991224, -9.07143945, 67.531)  # Negative X direction

    ECEF0 = geoTransformer.gps_to_ecef(*GPS0)
    ECEFX = geoTransformer.gps_to_ecef(*GPSX)
    ECEFY = geoTransformer.gps_to_ecef(*GPSY)

    R_LocalToGlobal = geoTransformer.get_rotation(ECEF0, ECEFX, ECEFY)
    points_local_cs[:, 0] *= -1  # Flip X for left-handed CS

    gps_points = []
    GPS_base_curr = (53.2902897942, -9.0711320615, 72.3162901802)
    ECEF_base_curr = geoTransformer.gps_to_ecef(*GPS_base_curr)

    # Convert points to GPS and save to list
    for i in tqdm(range(len(points_local_cs))):
        point_global_cs_ecef = geoTransformer.lidar_to_ecef(points_local_cs[i], ECEF_base_curr, R_LocalToGlobal)
        gps_point = geoTransformer.ecef_to_gps(*point_global_cs_ecef)

        gps_points.append([1, f"point_car_{i}", gps_point[0], gps_point[1], gps_point[2], 1])  # Colour_ID=1, Show=1

    # Define CSV output file
    output_file = "G:/Documents/Pycharm Projects/Work_Package_1/src/data_processing/gps_points_2.csv"

    # Write GPS data to CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Colour_ID", "Name", "Longitude", "Latitude", "Altitude", "Show"])  # Header
        writer.writerows(gps_points)  # Data content

    print(f"GPS data saved to {output_file}")

if __name__ == "__main__":
    main()