import pandas as pd

df = pd.read_csv("G:/Documents/Pycharm Projects/Work_Package_1/data/dangan_surveying_12-06-24/emlid_csv/merged_surveying_csv_emlid_app.csv")
# slot_1 = ['cpsl-1', 'cpsl-2', 'cpsl-3', 'cpsl-4']
# slot_2 = ['cpsl-3', 'cpsl-4', 'cpsl-5', 'cpsl-6']
# slot_4 = ['cpsl-5', 'cpsl-6', 'cpsl-7', 'cpsl-8']
# slot_5 = ['cpsl-7', 'cpsl-8', 'cpsl-9', 'cpsl-10']
# slot_6 = ['cpsl-9', 'cpsl-10', 'cpsl-13', 'cpsl-14']
# slot_7 = ['cpsl-13', 'cpsl-14', 'cpsl-15', 'cpsl-16']
# slot_8 = ['cpsl-15', 'cpsl-16', 'cpsl-17', 'cpsl-18']
# slot_9 = ['cpsl-17', 'cpsl-18', 'cpsl-23', 'cpsl-24']
# slots = [slot_1, slot_2, slot_4, slot_5, slot_6, slot_7, slot_8, slot_9]


slot_1 = ['cpslr-1', 'cpslr-2', 'cpslr-3', 'cpslr-4']
slot_2 = ['cpslr-3', 'cpslr-4', 'cpslr-5', 'cpslr-12']
slot_3 = ['cpslr-5', 'cpslr-12', 'cpslr-6', 'cpslr-13']
slot_4 = ['cpslr-6', 'cpslr-13', 'cpslr-7', 'cpslr-14']
slot_5 = ['cpslr-7', 'cpslr-14', 'cpslr-8', 'cpslr-15']
slot_6 = ['cpslr-8', 'cpslr-15', 'cpslr-9', 'cpslr-16']
slot_7 = ['cpslr-9', 'cpslr-16', 'cpslr-10', 'cpslr-17']
slot_8 = ['cpslr-10', 'cpslr-17', 'cpslr-11', 'cpslr-18']
slot_9 = ['cpslr-11', 'cpslr-18', 'cpslr-19', 'cpslr-20']
slot_10 = ['cpslr-19', 'cpslr-20', 'cpslr-21', 'cpslr-22']
slot_11 = ['cpslr-21', 'cpslr-22', 'cpslr-23', 'cpslr-24']
slot_12 = ['cpslr-23', 'cpslr-24', 'cpslr-25', 'cpslr-26']
slot_13 = ['cpslr-27', 'cpslr-28', 'cpslr-29', 'cpslr-30']
slot_14 = ['cpslr-31', 'cpslr-32', 'cpslr-33', 'cpslr-48']
slot_15 = ['cpslr-34', 'cpslr-49', 'cpslr-35', 'cpslr-50']
slot_16 = ['cpslr-36', 'cpslr-51', 'cpslr-37', 'cpslr-52']
slot_17 = ['cpslr-38', 'cpslr-53', 'cpslr-39', 'cpslr-54']
slot_18 = ['cpslr-39', 'cpslr-54', 'cpslr-40', 'cpslr-55']
slot_19 = ['cpslr-40', 'cpslr-55', 'cpslr-41', 'cpslr-56']
slot_20 = ['cpslr-41', 'cpslr-56', 'cpslr-42', 'cpslr-57']
slot_21 = ['cpslr-42', 'cpslr-57', 'cpslr-43', 'cpslr-58']
slot_22 = ['cpslr-43', 'cpslr-58', 'cpslr-44', 'cpslr-59']
slot_23 = ['cpslr-44', 'cpslr-59', 'cpslr-45', 'cpslr-60']
slot_24 = ['cpslr-45', 'cpslr-60', 'cpslr-46', 'cpslr-61']
slot_25 = ['cpslr-46', 'cpslr-61', 'cpslr-47', 'cpslr-62']

slots = [slot_1, slot_2, slot_3, slot_4, slot_5, slot_6, slot_7, slot_8, slot_9, slot_10, slot_11, slot_12, slot_13, slot_14,
         slot_15, slot_16, slot_17, slot_18, slot_19, slot_20, slot_21, slot_22, slot_23, slot_24, slot_25]

center_points = {}
names_to_keep = [name for sublist in slots for name in sublist]
df = df[df['Name'].isin(names_to_keep)]

def get_center_point(names):
    df_slot = df[df['Name'].isin(names)]
    center_longitude = df_slot['Longitude'].mean()
    center_latitude = df_slot['Latitude'].mean()
    center_height = df_slot['Ellipsoidal height'].mean()
    return center_longitude, center_latitude, center_height

for i, slot in enumerate(slots, 1):
    # slot_name = 'cpsl_' + slot[0].replace('cpsl-', '') + '-' + '-'.join([name.replace('cpsl-', '') for name in slot[1:]]) + '_ctr'
    slot_name = 'cpslr_'+slot[0].replace('cpslr-', '') + '-' + '-'.join([name.replace('cpslr-', '') for name in slot[1:]]) + '_ctr'
    center_points = get_center_point(slot)
    print(f"{slot_name}, {center_points[0]}, {center_points[1]}, {center_points[2]}")

