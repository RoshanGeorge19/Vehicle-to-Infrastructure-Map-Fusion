import pandas as pd

def main():
    df_car_obj = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/proc/car/person_1/id-307_id-356_person_1_annotation_w_gps_min_4_shift.csv")
    df_node_obj = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/processed_csv_files/annotations/raw/node/id-259_id-308_person_1_annotations_node.csv")
    df_node_obj = df_node_obj.rename(columns={'Time_Short': 'Time_Short_node'})
    df_merged = pd.merge_asof(df_car_obj.sort_values('Time_Short'), df_node_obj.sort_values('Time_Short_node'),
                              left_on='Time_Short', right_on='Time_Short_node', direction='backward')

    df_merged = df_merged.drop(
        columns=['Rx_x', 'Ry_x', 'Rz_x', 'Time_y', 'Rx(Roll)', 'Ry(Pitch)', 'Rz(Yaw)', 'X', 'Y', 'Z', 'Shifted_Time',
                 'Rx_y', 'Ry_y', 'Rz_y'])

    print(df_merged.head())


if __name__ == '__main__':
    main()