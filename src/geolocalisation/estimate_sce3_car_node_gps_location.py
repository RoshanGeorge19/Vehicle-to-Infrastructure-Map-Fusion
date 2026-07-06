# T1 and T2 are translation vectors.
# T1 and T2 are gotten from the lidar registration process.
# T1 = G:\Documents\Pycharm Projects\Work_Package_1\src\matlab_processing\edge_map_align_car_vlp.m
# T2 = G:\Documents\Pycharm Projects\Work_Package_1\src\matlab_processing\car_vlp_align_node_vlp.m
# For pre-proc look at G:\Documents\Pycharm Projects\Work_Package_1\src\matlab_processing\rotate_pcd_lidar_to_local_cs
# Data available at: G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\CAR\Scenario-3\point_cloud_edited_alignment_node_car\
# Data for node vlp, car vlp, and edge_map, have been cropped using matlab-lidarViewer. This allows for better registration.

import numpy as np
from wp2.geo_utils import GeoTransformer
geo_transformer = GeoTransformer()

T1 = np.asarray([1.5199282, 68.813316, 7.3688855]) # Edge Map to Velo Car.
T2 = np.asarray([3.8249, 70.6997, 7.3680]) # Velo Car to Velo Node

GPS0 = (53.28989834, -9.07136142, 67.566)
GPSY = (53.28991859, -9.07135091, 67.552)
GPSX = (53.28991224, -9.07143945, 67.531)
GPS_start = (53.28988778238876, -9.071366899526426, 67.57329943403602)

ECEF0 = geo_transformer.gps_to_ecef(*GPS0)
ECEFX = geo_transformer.gps_to_ecef(*GPSX)
ECEFY = geo_transformer.gps_to_ecef(*GPSY)
R_local_to_global = geo_transformer.get_rotation(ECEF0, ECEFX, ECEFY)
ECEF_start = geo_transformer.gps_to_ecef(*GPS_start)

lidar_point_1 = np.array([-T1[0], T1[1], T1[2]])
ecef_point_1 = geo_transformer.lidar_to_ecef(lidar_point_1, ECEF_start, R_local_to_global)
gps_point_1 = geo_transformer.ecef_to_gps(*ecef_point_1)
print(f"Car Estimated Location ({gps_point_1[0]}, {gps_point_1[1]})")

T3 = T1 + T2 # Edge Map tp Velo Car + Velo Car to Velo Node
lidar_point_2 = np.array([-T3[0], T3[1], T3[2]])
ecef_point_2 = geo_transformer.lidar_to_ecef(lidar_point_2, ECEF_start, R_local_to_global)
gps_point_2 = geo_transformer.ecef_to_gps(*ecef_point_2)
print(f"Node Estimated Location ({gps_point_2[0]}, {gps_point_2[1]})")

T5 = np.asarray([3.75856, 70.77945, 7.36016])
lidar_point_1 = np.array([-T5[0], T5[1], T5[2]])
ecef_point_1 = geo_transformer.lidar_to_ecef(lidar_point_1, ECEF_start, R_local_to_global)
gps_point_1 = geo_transformer.ecef_to_gps(*ecef_point_1)
print(f"Estimated Location ({gps_point_1[0]}, {gps_point_1[1]})")