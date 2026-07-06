"""
Extends the surveyed parking-spot corner grid outward by the average spot
spacing, to estimate the GPS location of corners that weren't directly
surveyed.

Merged from extend_corners_car_parking_spots.py (top-left corner) and
extend_corners_car_parking_spots_bottom_right.py (bottom-right corner), which
differed only in which surveyed corner they started from, the sign of the
X offset, and how many grid steps to extend. Run with `--corner top-left`
or `--corner bottom-right`.
"""
import argparse
import statistics

import nvector as nv
import numpy as np
import pandas as pd

from wp2.geo_utils import GeoTransformer

CORNER_CONFIGS = {
    'top-left': {
        'base_spots': ['cpsl-17', 'cpsl-24', 'cpsl-23'],
        'x_sign': 1,
        'num_steps': 13,
        'label': 'gps_new',
    },
    'bottom-right': {
        'base_spots': ['csl-18', 'csl-19', 'csl-8'],
        'x_sign': -1,
        'num_steps': 12,
        'label': 'gps_new_bottom_right',
    },
}


def get_mean_distance(df):
    y_points = ['cpsl-1', 'cpsl-4', 'cpsl-5', 'cpsl-8', 'cpsl-9', 'cpsl-13', 'cpsl-16', 'cpsl-17', 'cpsl-24']
    y_points_dict = {point: (df.loc[df['Name'] == point, 'Longitude'].values[0], df.loc[df['Name'] == point, 'Latitude'].values[0]) for point in y_points}

    x_points = ['cpsl-1', 'cpsl-2', 'cpsl-3', 'cpsl-4', 'cpsl-5', 'cpsl-6', 'cpsl-7', 'cpsl-8', 'cpsl-9', 'cpsl-10', 'cpsl-13', 'cpsl-14', 'cpsl-15', 'cpsl-16', 'cpsl-17', 'cpsl-18', 'cpsl-23', 'cpsl-24']
    x_points_dict = {point: (df.loc[df['Name'] == point, 'Longitude'].values[0], df.loc[df['Name'] == point, 'Latitude'].values[0]) for point in x_points}

    wgs84 = nv.FrameE(name='WGS84')
    distances_euclidean_ecef_y = {}
    distances_euclidean_ecef_x = {}

    for i in range(len(y_points) - 1):
        pointA_name, pointB_name = y_points[i], y_points[i + 1]
        lonA, latA = y_points_dict[pointA_name]
        lonB, latB = y_points_dict[pointB_name]
        pointA = wgs84.GeoPoint(latitude=latA, longitude=lonA, degrees=True)
        pointB = wgs84.GeoPoint(latitude=latB, longitude=lonB, degrees=True)
        distances_euclidean_ecef_y[(pointA_name, pointB_name)] = (pointB.to_ecef_vector() - pointA.to_ecef_vector()).length

    for ii in range(0, len(x_points) - 1, 2):
        pointA_name, pointB_name = x_points[ii], x_points[ii + 1]
        lonA, latA = x_points_dict[pointA_name]
        lonB, latB = x_points_dict[pointB_name]
        pointA = wgs84.GeoPoint(latitude=latA, longitude=lonA, degrees=True)
        pointB = wgs84.GeoPoint(latitude=latB, longitude=lonB, degrees=True)
        distances_euclidean_ecef_x[(pointA_name, pointB_name)] = (pointB.to_ecef_vector() - pointA.to_ecef_vector()).length

    distances_list_y = list(distances_euclidean_ecef_y.values())
    distances_list_x = list(distances_euclidean_ecef_x.values())

    mean_distance_y = statistics.mean(distances_list_y[1:])
    mean_distance_x = statistics.mean(distances_list_x[0:])

    return mean_distance_y, mean_distance_x


def calculate_new_gps_points(geo_transformer, lonBase, latBase, lonX, latX, lonY, latY, mean_distance_x, mean_distance_y, x_sign, R):
    ECEF_Y = geo_transformer.gps_to_ecef(latY, lonY, 0)

    new_y = np.array([0, mean_distance_y, 0])
    new_x = np.array([x_sign * mean_distance_x, mean_distance_y, 0])

    ecef_new_y = geo_transformer.lidar_to_ecef(new_y, ECEF_Y, R)
    gps_new_y = geo_transformer.ecef_to_gps(*ecef_new_y)

    ecef_new_x = geo_transformer.lidar_to_ecef(new_x, ECEF_Y, R)
    gps_new_x = geo_transformer.ecef_to_gps(*ecef_new_x)

    return gps_new_y, gps_new_x


def getRotationMatrixFromGPS(geo_transformer, lonBase, latBase, lonX, latX, lonY, latY):
    ECEF_Base = geo_transformer.gps_to_ecef(latBase, lonBase, 0)
    ECEF_Y = geo_transformer.gps_to_ecef(latY, lonY, 0)
    ECEF_X = geo_transformer.gps_to_ecef(latX, lonX, 0)
    return geo_transformer.get_rotation(ECEF_Base, ECEF_X, ECEF_Y)


def main(corner):
    config = CORNER_CONFIGS[corner]
    geo_transformer = GeoTransformer()

    df = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/dangan_surveying_12-06-24/emlid_csv/merged_surveying_csv_emlid_app.csv")
    mean_distance_y, mean_distance_x = get_mean_distance(df)

    base_spot, y_spot, x_spot = config['base_spots']
    corner_basexy_spots_dict = {
        point: (df.loc[df['Name'] == point, 'Longitude'].values[0], df.loc[df['Name'] == point, 'Latitude'].values[0])
        for point in config['base_spots']
    }

    base_points = [corner_basexy_spots_dict[base_spot]]
    y_points = [corner_basexy_spots_dict[y_spot]]
    x_points = [corner_basexy_spots_dict[x_spot]]

    for i in range(config['num_steps']):
        lonBase, latBase = base_points[0]
        lonY, latY = y_points[0]
        lonX, latX = x_points[0]

        R = getRotationMatrixFromGPS(geo_transformer, lonBase, latBase, lonX, latX, lonY, latY)
        gps_new_y, gps_new_x = calculate_new_gps_points(
            geo_transformer, lonBase, latBase, lonX, latX, lonY, latY, mean_distance_x, mean_distance_y, config['x_sign'], R)

        print(f"10, {config['label']}_y_{i},{gps_new_y[0]}, {gps_new_y[1]}, {gps_new_y[2]},1")
        print(f"10, {config['label']}_x_{i},{gps_new_x[0]}, {gps_new_x[1]}, {gps_new_x[2]},1")

        base_points = [(lonY, latY)]
        y_points = [(gps_new_y[0], gps_new_y[1])]
        x_points = [(gps_new_x[0], gps_new_x[1])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corner', choices=['top-left', 'bottom-right'], required=True, help='Which surveyed corner to extend the grid from')
    args = parser.parse_args()
    main(args.corner)
