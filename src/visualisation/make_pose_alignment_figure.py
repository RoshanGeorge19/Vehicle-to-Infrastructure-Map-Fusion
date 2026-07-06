import csv
import laspy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm
from wp2.geo_utils import GeoTransformer

node_geolocate_object = GeoTransformer.node_geolocate_object

def plot_lidar_data(x, y, z, intensity, annotation):
    fig, ax = plt.subplots()
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    x_center = annotation['X_Center']
    y_center = annotation['Y_Center']
    length = 0.7
    width = 0.5
    name = annotation['File_Name']

    # Compute bounding box limits
    x_min = x_center - length / 2
    x_max = x_center + length / 2
    y_min = y_center - width / 2
    y_max = y_center + width / 2

    # Identify points inside the bounding box
    inside_bbox_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    if 'car' in name.lower():
        pt_color = 'green'
    elif 'cepton' in name.lower():
        pt_color = 'blue'

    ax.scatter(x[inside_bbox_mask], y[inside_bbox_mask], c='black', s=2.5, label='Object Points')
    sc = ax.scatter(x[~inside_bbox_mask], y[~inside_bbox_mask], c=pt_color, s=2.5)
    bbox = patches.Rectangle((x_min, y_min), length, width, linewidth=1.5, edgecolor='black', facecolor='none')

    ax.add_patch(bbox)
    plt.xlabel('X-Axis (m)')
    plt.ylabel('Y-Axis (m)')
    plt.grid(True)
    # plt.legend()
    plt.show()

def main():
    geoTransformer = GeoTransformer()
    car_las_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/car_cepton_overlap_1/las_out_id-307_id-356/1099.700-Car_Scene-2.las"
    car_annotations_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_1_annotations_car.csv"
    node_las_path = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/Cepton/Scenario2-1/car_cepton_overlap_1/las_out_id-259_id-308/1099.644-Cepton_Scene-2.las'
    node_annotations_path = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_1_annotations_node.csv'

    car_annotations = pd.read_csv(car_annotations_path)
    node_annotations = pd.read_csv(node_annotations_path)
    car_annotation = car_annotations.iloc[0]
    node_annotation = node_annotations.iloc[0]

    node_las = laspy.read(node_las_path)
    node_points = (node_las.x, node_las.y, node_las.z)
    node_x, node_y, node_z = node_points
    node_intensity = node_las.intensity
    mask = (np.abs(node_x) <= 10) & (np.abs(node_y) <= 20)
    node_x_mask, node_y_mask, node_z_mask = node_x[mask], node_y[mask], node_z[mask]
    node_intensity_mask = node_intensity[mask]

    car_las = laspy.read(car_las_path)
    car_points = (car_las.x, car_las.y, car_las.z)
    car_x, car_y, car_z = car_points
    car_intensity = car_las.intensity
    mask = (np.abs(car_x) <= 10) & (np.abs(car_y) <= 20)
    car_x_mask, car_y_mask, car_z_mask = car_x[mask], car_y[mask], car_z[mask]
    car_intensity_mask = car_intensity[mask]

    plot_lidar_data(node_x_mask, node_y_mask, node_z_mask, node_intensity_mask, node_annotation)
    plot_lidar_data(car_x_mask, car_y_mask, car_z_mask, car_intensity_mask, car_annotation)

    output_file = "G:/Documents/Pycharm Projects/Work_Package_1/src/data_processing/gps_points.csv"

    compass_heading = 245.1
    node_GPS = (53.29047403, -9.07095837)
    gps_points_node = []

    coords = np.vstack((node_x_mask, node_y_mask, node_z_mask)).T
    for i in tqdm(range(len(coords))):
        gps_point = node_geolocate_object(node_GPS, compass_heading, coords[i])
        gps_points_node.append([1, f"node_{i}", gps_point[0], gps_point[1], gps_point[2], 1])  # Colour_ID=1, Show=1


    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Colour_ID", "Name", "Longitude", "Latitude", "Altitude", "Show"])  # Header
        writer.writerows(gps_points_node)  # Data content

    theta = np.deg2rad(-15)
    R_LidarToLocal = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    coords = np.vstack((car_x_mask, car_y_mask, car_z_mask))
    points_local_cs = np.dot(R_LidarToLocal, coords).T

    GPS0 = (53.28989834, -9.07136142, 67.566)  # Origin
    GPSY = (53.28991859, -9.07135091, 67.552)  # Positive Y direction
    GPSX = (53.28991224, -9.07143945, 67.531)  # Negative X direction
    ECEF0 = geoTransformer.gps_to_ecef(*GPS0)
    ECEFX = geoTransformer.gps_to_ecef(*GPSX)
    ECEFY = geoTransformer.gps_to_ecef(*GPSY)
    R_LocalToGlobal = geoTransformer.get_rotation(ECEF0, ECEFX, ECEFY)
    points_local_cs[:, 0] *= -1  #Flip X for left-handed CS
    gps_points_car = []
    GPS_base_curr = (53.2902897942, -9.0711320615, 72.3162901802)
    ECEF_base_curr = geoTransformer.gps_to_ecef(*GPS_base_curr)

    # Convert points to GPS and save to list
    for i in tqdm(range(len(points_local_cs))):
        point_global_cs_ecef = geoTransformer.lidar_to_ecef(points_local_cs[i], ECEF_base_curr, R_LocalToGlobal)
        gps_point = geoTransformer.ecef_to_gps(*point_global_cs_ecef)
        gps_points_car.append([2, f"car_{i}", gps_point[0], gps_point[1], gps_point[2], 1])  # Colour_ID=1, Show=1


    # Write GPS data to CSV
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(gps_points_car)  # Data content

    print(f"GPS data saved to {output_file}")


if __name__ == "__main__":
    main()