import nvector as nv
from wp2.geo_utils import GeoTransformer
import numpy as np

geo_transformer = GeoTransformer()

wgs84 = nv.FrameE(name='WGS84')
csl4 = (53.28989582, -9.07141370, 67.612)
csl8 = (53.28991676, -9.07140185, 67.575)

pointA = wgs84.GeoPoint(latitude=csl4[0], longitude=csl4[1], z=csl4[2], degrees=True)
pointB = wgs84.GeoPoint(latitude=csl8[0], longitude=csl8[1], z=csl8[2], degrees=True)
pointBA_ECEF_vector = pointB.to_ecef_vector() - pointA.to_ecef_vector()
distance_euclidean_ecef = pointBA_ECEF_vector.length
delta = distance_euclidean_ecef/2

GPS0 = (53.28989834, -9.07136142, 67.566)  # Origin
GPSY = (53.28991859, -9.07135091, 67.552)  # Positive Y direction
GPSX = (53.28991224, -9.07143945, 67.531)  # Negative X direction

ECEF0 = geo_transformer.gps_to_ecef(*GPS0)
ECEFX = geo_transformer.gps_to_ecef(*GPSX)
ECEFY = geo_transformer.gps_to_ecef(*GPSY)

R = geo_transformer.get_rotation(ECEF0, ECEFX, ECEFY)

lidar_point = np.array([0, -delta, 0])
ecef_point = geo_transformer.lidar_to_ecef(lidar_point, ECEF0, R)
gps_point = geo_transformer.ecef_to_gps(*ecef_point)

print(f"1, {gps_point[0]}, {gps_point[1]}, {gps_point[2]}")