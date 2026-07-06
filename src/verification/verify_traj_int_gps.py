from wp2.geo_utils import GeoTransformer
import nvector as nv

geo_transformer = GeoTransformer()

GT = (53.2903625, -9.07117803, 67.092)
t1 = (53.29031901, -9.071114223, 72.53651248)
t2 = (53.29032198, -9.071112578, 72.57511998)
t3 = (53.29032532, -9.071110884, 72.62035791)
t4 = (53.29032576, -9.07111029, 67.078)
t5 = (53.29032868, -9.071108842, 72.66029975)
t6 = (53.29033171, -9.071106844, 72.69611427)
t7 = (53.2903349, -9.071105001, 72.72800317)

ECEF_GT = geo_transformer.gps_to_ecef(*GT)
ECEF_T1 = geo_transformer.gps_to_ecef(*t1)
ECEF_T2 = geo_transformer.gps_to_ecef(*t2)
ECEF_T3 = geo_transformer.gps_to_ecef(*t3)
ECEF_T4 = geo_transformer.gps_to_ecef(*t4)
ECEF_T5 = geo_transformer.gps_to_ecef(*t5)
ECEF_T6 = geo_transformer.gps_to_ecef(*t6)
ECEF_T7 = geo_transformer.gps_to_ecef(*t7)

# print(ECEF_T1)
# print(ECEF_T2)
# print(ECEF_T3)
# print(ECEF_T4)
# print(ECEF_T5)
# print(ECEF_T6)
# print(ECEF_T7)

wgs84 = nv.FrameE(name='WGS84')
distances_great_circle_y = {}
distances_great_circle_x = {}

distances_euclidean_ecef_y = {}
distances_euclidean_ecef_x = {}

base = t7

pointA = wgs84.GeoPoint(latitude=base[0], longitude=base[1], degrees=True)
pointB = wgs84.GeoPoint(latitude=GT[0], longitude=GT[1], degrees=True)
distance_great_circle_ab, _, _ = pointA.distance_and_azimuth(pointB)

pointBA_ECEF_vector = pointB.to_ecef_vector() - pointA.to_ecef_vector()
distance_euclidean_ecef = pointBA_ECEF_vector.length

print(distance_euclidean_ecef)
