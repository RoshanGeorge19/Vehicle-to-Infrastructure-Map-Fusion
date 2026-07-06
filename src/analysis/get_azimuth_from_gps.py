import nvector as nv
import pandas as pd
from nvector import rad, deg
import geopy.distance
import math

df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/id-307_id-356_person_1_annotation_with_gps.csv')
wgs84 = nv.FrameE(name='WGS84')

df['Next Longitude'] = df['Base Longitude'].shift(-1)
df['Next Latitude'] = df['Base Latitude'].shift(-1)
df['Next Altitude'] = df['Base Altitude'].shift(-1)


def geolocate_object(lidar_origin_gps, compass_heading, object_lidar_coords):
    # Unpack the input tuples
    lat0, lon0 = lidar_origin_gps
    x, y, z = object_lidar_coords
    x = -x

    # Convert compass heading to radians
    heading_rad = math.radians(compass_heading)

    # Compute global offsets
    # Forward (Y in LiDAR) corresponds to North-South direction
    north_offset = y * math.cos(heading_rad) + x * math.sin(heading_rad)
    east_offset = y * math.sin(heading_rad) - x * math.cos(heading_rad)

    # Convert offsets from meters to GPS coordinates
    # Create a geopy point for the origin
    origin = geopy.Point(lat0, lon0)

    # Calculate the new location based on offsets
    north_distance = geopy.distance.distance(meters=north_offset)
    east_distance = geopy.distance.distance(meters=east_offset)

    # Move north and then east
    new_location = north_distance.destination(point=origin, bearing=0)
    new_location = east_distance.destination(point=new_location, bearing=90)

    # Return the GPS coordinates of the detected object
    return new_location.latitude, new_location.longitude

for index, row in df.iterrows():
    lon_0 = row['Base Longitude']
    lat_0 = row['Base Latitude']
    alt_0 = row['Base Altitude']
    lon_1 = row['Next Longitude']
    lat_1 = row['Next Latitude']
    alt_1 = row['Next Altitude']

    pointA = wgs84.GeoPoint(latitude=lat_0, longitude=lon_0, z=alt_0, degrees=True)
    pointB = wgs84.GeoPoint(latitude=lat_1, longitude=lon_1, z=alt_1, degrees=True)

    pointA_delta_to_pointB = pointA.delta_to(pointB)
    azimuth = pointA_delta_to_pointB.azimuth_deg
    azimuth = 360 - azimuth
    # print(f"The azimuth between the points is {azimuth} degrees.")

    lidar_origin_gps = (row['Base Latitude'], row['Base Longitude'])  # Example GPS coordinates (latitude, longitude)
    object_lidar_coord = (row['X_Center'], row['Y_Center'], row['Z_Center'])
    object_gps = geolocate_object(lidar_origin_gps, azimuth, object_lidar_coord)
    print(f"{object_gps[1]}, {object_gps[0]}")



