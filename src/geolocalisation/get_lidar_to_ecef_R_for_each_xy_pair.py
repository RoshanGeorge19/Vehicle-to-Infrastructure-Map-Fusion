import pandas as pd
from wp2.geo_utils import GeoTransformer
import numpy as np

def main():
    geo_transformer = GeoTransformer()
    df_xy_names = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/x_y_gps_pair_names_for_R_local_to_ecef.csv')
    df_gps = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/gps_points_for_x_y_pairs_to_get_R_local_to_ecef.csv')

    df_xy_names_w_R = df_xy_names.copy()
    rotation_vec = []
    for index, row in df_xy_names.iterrows():
        origin = df_gps.loc[df_gps['Name'] == row['Base_Point'], ['Longitude', 'Latitude', 'Altitude']]
        y_axis_gps_pt = df_gps.loc[df_gps['Name'] == row['Y_Axis_Name'], ['Longitude', 'Latitude', 'Altitude']]
        x_axis_gps_pt = df_gps.loc[df_gps['Name'] == row['X_Axis_Name'], ['Longitude', 'Latitude', 'Altitude']]
        print(f"{row['Base_Point']}, {row['Y_Axis_Name']}, {row['X_Axis_Name']}")

        ECEF0 = geo_transformer.gps_to_ecef(*(origin.values[0][1], origin.values[0][0], origin.values[0][2]))
        ECEFY = geo_transformer.gps_to_ecef(*(y_axis_gps_pt.values[0][1], y_axis_gps_pt.values[0][0], y_axis_gps_pt.values[0][2]))
        ECEFX = geo_transformer.gps_to_ecef(*(x_axis_gps_pt.values[0][1], x_axis_gps_pt.values[0][0], x_axis_gps_pt.values[0][2]))
        R = geo_transformer.get_rotation(ECEF0, ECEFX, ECEFY)
        R = R.flatten()
        R_str = np.array2string(R, separator=',').replace('\n', '')
        rotation_vec.append(R_str)

    rotation_matrices_series = pd.Series(rotation_vec)
    df_xy_names_w_R['R'] = rotation_matrices_series
    df_xy_names_w_R.to_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/x_y_gps_pair_names_with_R_for_local_to_ecef.csv', index=False)

if __name__ == '__main__':
    main()