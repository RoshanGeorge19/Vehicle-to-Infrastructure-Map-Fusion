# Merges trajectory and gps information based on time.
import pandas as pd

# Specify the file path and sheet name
file_path = '/data/lidar_data/CAR/Scenario-2/Data Processing.xlsx'

# Read the Excel file
df = pd.read_excel(file_path, sheet_name='Slam Trajectory 200-400')

# Create a new DataFrame with the specified columns
df_new = df[['Time', 'Rx(Roll)', 'Ry(Pitch)', 'Rz(Yaw)', 'X', 'Y', 'Z']]

# Create a new DataFrame with the specified columns
df_new = df[['Time', 'Rx(Roll)', 'Ry(Pitch)', 'Rz(Yaw)', 'X', 'Y', 'Z']].copy()

# The rest of your code remains the same

# Round the 'Time' column in df_new and 'Car_Matched_GPS-Drift' column in df to two decimal places
df_new['Time'] = df_new['Time'].round(2)
df['Car_Matched_GPS - Drift'] = df['Car_Matched_GPS - Drift'].round(2)

# Merge df_new and df on the rounded columns, using a left join to keep all rows from df_new
df_new = pd.merge(df_new, df[['Car_Matched_GPS - Drift', 'Latitude', 'Longitude', 'Altitude', 'Index']],
                  left_on='Time', right_on='Car_Matched_GPS - Drift', how='left')

# Drop the 'Car_Matched_GPS-Drift' column as it's no longer needed
df_new = df_new.drop(columns=['Car_Matched_GPS - Drift'])

# Replace NaN values in the 'Latitude', 'Longitude', 'Altitude', and 'Index' columns with 0
df_new[['Latitude', 'Longitude', 'Altitude', 'Index']] = df_new[['Latitude', 'Longitude', 'Altitude', 'Index']].fillna(0)
df_new['Time'] = df['Time']

# Specify the output file path
output_file_path = '../../../data/lidar_data/CAR/Dangan GPS Data - 2022-05-27/Traj_and_GPS_merged.csv'

# Save the DataFrame as a CSV file
df_new.to_csv(output_file_path, index=False)
