import pandas as pd
from datetime import datetime

# Read the CSV file
df = pd.read_csv('/data/lidar_data/CAR/Dangan GPS Data - 2022-05-27/2022-05-27-Scenario-3-GPS.csv')
print(df.columns)
# Remove the unwanted columns
df = df.drop(columns=['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'])

# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Subtract an hour from the 'Time' column
df['Time_Fixed'] = df['Time'] - pd.Timedelta(hours=1)

# Convert the datetime column to Unix timestamp in microseconds and assign it to a new column 'UnixTime'
df['Time_Fixed(ms)'] = (df['Time_Fixed'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

# Rearrange the columns
df = df[['Unnamed: 0', 'Time', 'Time(ms)', 'Time_Fixed', 'Time_Fixed(ms)', 'Latitude', 'Longitude', 'Altitude', 'Bearing', 'Speed', 'Climb']]

# Save the dataframe to a new CSV file
df.to_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Dangan GPS Data - 2022-05-27/Scenario-3-GPS_Modified.csv', index=False)
