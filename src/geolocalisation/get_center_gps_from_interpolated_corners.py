import pandas as pd
# CSV contains name, long, lat, alt. Outputted from extend_corners_car_parking_spots.py.
# e.g., gps_new_x_i, longitude, latitude, altitude
# e.g., gps_new_y_i, longitude, latitude, altitude
# df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/gps_predicted_corners_left.csv')

# CSV contains name, long, lat, alt. Manual. Just needed a center point for cpsl 23 cpsl 24 x0 y0
df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/csv/gps_predicted_corners_left_csl1516_x11y11.csv')

# Create a new DataFrame where each row contains the data for a group of 4 rows from the original DataFrame
# i.e., index [0 1 2 3], [2 3 4 5], [4 5 6 7], ...
grouped_data = [df[i:i+4] for i in range(0, len(df)-3, 2)]

centers = []
for group in grouped_data:
    center_longitude = group['Longitude'].mean()
    center_latitude = group['Latitude'].mean()
    center_altitude = group['Altitude'].mean()
    centers.append((center_longitude, center_latitude, center_altitude))

# Print the center points
for center in centers:
    print(f"{center[0]}, {center[1]}, {center[2]}")