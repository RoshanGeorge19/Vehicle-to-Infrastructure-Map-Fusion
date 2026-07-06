# For emlid, merges the csv files into one. It also renames stuff for offline analysis.
import pandas as pd

# Set to True if you want to modify the csv files for offline analysis. Set false for visualisation purpose only using Emlid Flow.
modify_for_analysis = False

df_car_parking_spots_left = pd.read_csv("/data/Dangan_Surveying_12-06-24/car parking spots left.csv")
df_car_parking_spots_right = pd.read_csv("/data/Dangan_Surveying_12-06-24/car parking spots right.csv")
df_car_start_far_left = pd.read_csv("/data/Dangan_Surveying_12-06-24/car start - far left.csv")
df_road_arrows = pd.read_csv("/data/Dangan_Surveying_12-06-24/road arrows.csv")
df_static_signs = pd.read_csv("/data/Dangan_Surveying_12-06-24/static signs.csv")
df_wooden_posts = pd.read_csv("/data/Dangan_Surveying_12-06-24/wooden posts.csv")


def add_filename_column(df, filename):
    df.insert(0, 'File Name', filename)

if modify_for_analysis:
    add_filename_column(df_car_parking_spots_left, 'car_parking_spots_left')
    add_filename_column(df_car_parking_spots_right, 'car_parking_spots_right')
    add_filename_column(df_car_start_far_left, 'car_start_far_left')
    add_filename_column(df_road_arrows, 'road_arrows')
    add_filename_column(df_static_signs, 'static_signs')
    add_filename_column(df_wooden_posts, 'wooden_posts')
else:
    add_filename_column(df_car_parking_spots_left, 'cpsl')
    add_filename_column(df_car_parking_spots_right, 'cpsr')
    add_filename_column(df_car_start_far_left, 'csl')
    add_filename_column(df_road_arrows, 'ra')
    add_filename_column(df_static_signs, 'ss')
    add_filename_column(df_wooden_posts, 'wp')

columns_to_drop = ['Code', 'Easting', 'Northing', 'Elevation', 'Base easting', 'Base northing', 'Base elevation', 'Mount point']
satellite_columns = ['GPS Satellites', 'GLONASS Satellites', 'Galileo Satellites', 'BeiDou Satellites', 'QZSS Satellites']

if modify_for_analysis:
    df_car_parking_spots_left = df_car_parking_spots_left.drop(columns=columns_to_drop)
    df_car_parking_spots_right = df_car_parking_spots_right.drop(columns=columns_to_drop)
    df_car_start_far_left = df_car_start_far_left.drop(columns=columns_to_drop)
    df_road_arrows = df_road_arrows.drop(columns=columns_to_drop)
    df_static_signs = df_static_signs.drop(columns=columns_to_drop)
    df_wooden_posts = df_wooden_posts.drop(columns=columns_to_drop)

    df_car_parking_spots_left = df_car_parking_spots_left.assign(Total_Satellites=df_car_parking_spots_left[satellite_columns].sum(axis=1))
    df_car_parking_spots_right = df_car_parking_spots_right.assign(Total_Satellites=df_car_parking_spots_right[satellite_columns].sum(axis=1))
    df_car_start_far_left = df_car_start_far_left.assign(Total_Satellites=df_car_start_far_left[satellite_columns].sum(axis=1))
    df_road_arrows = df_road_arrows.assign(Total_Satellites=df_road_arrows[satellite_columns].sum(axis=1))
    df_static_signs = df_static_signs.assign(Total_Satellites=df_static_signs[satellite_columns].sum(axis=1))
    df_wooden_posts = df_wooden_posts.assign(Total_Satellites=df_wooden_posts[satellite_columns].sum(axis=1))
else:
    pass

merged_df = pd.concat([df_car_parking_spots_left, df_car_parking_spots_right, df_car_start_far_left, df_road_arrows, df_static_signs, df_wooden_posts])

if modify_for_analysis:
    save_file_name = "/data/Dangan_Surveying_12-06-24/merged_surveying_csv.csv"

else:
    save_file_name = "/data/Dangan_Surveying_12-06-24/merged_surveying_csv_emlid_app.csv"
    merged_df['Name'] = merged_df.apply(lambda row: f"{row['File Name']}-{row['Name']}", axis=1)
    merged_df = merged_df.drop(columns='File Name')

merged_df.to_csv(save_file_name, index=False)