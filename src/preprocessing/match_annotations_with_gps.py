# For the annotations and the given time. It finds the closest GPS point and adds the GPS information to the annotations.
import pandas as pd

df1 = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/car/id-307_id-356_person_3_annotations_car.csv")
df2 = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/slam/v2/trajectory_v2_global_cs.csv")

# Round the 'Time' column in both dataframes to ensure proper matching
df1['Time_Short'] = df1['Time_Short'].round(2)
df2['Time'] = df2['Time'] - 0.05
df2['Time'] = df2['Time'].round(2)

# df2['Shifted_Time'] = df2['Time'].shift(-4)
df2['Shifted_Time'] = df2['Time'].shift(0)
df3 = pd.merge(df1, df2, left_on='Time_Short', right_on='Shifted_Time', how='left')

# df3 = df1.copy()
# df3 = pd.merge(df3, df2, left_on='Time_Short', right_on='Time', how='left')

# df3.to_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/proc/car/person_3/id-307_id-356_person_3_annotations_w_gps_min_4_shift.csv", index=False)
print(df3)
