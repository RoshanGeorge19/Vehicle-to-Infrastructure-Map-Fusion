# Calls in two sheets from the excel file 'data processing'
# It then merges on the Time column so that the its matched on time.
# Then on the merged df it adds in bearing, climb and speed.
import pandas as pd

file_path = '/data/lidar_data/CAR/Scenario-2/Data Processing.xlsx'
df = pd.read_excel(file_path, sheet_name='GPS_Trajectory_Merged')
df2 = pd.read_excel(file_path, sheet_name='Car_307-356_GT_GPS')

# Round 'Time' column to two decimal places in both dataframes
df['Time'] = df['Time'].round(2)
df2['Time'] = df2['Time'].round(2)

# Merge df and df2 on 'Time'
merged_df = pd.merge(df2, df[['Time', 'Bearing', 'Climb', 'Speed']], on='Time', how='left')

df2 = pd.read_excel(file_path, sheet_name='Car_307-356_GT_GPS')
merged_df['Time'] = df2['Time']

# Print the merged dataframe with full time
merged_df.to_csv("add_bearing_and_climb.csv", index=False)
