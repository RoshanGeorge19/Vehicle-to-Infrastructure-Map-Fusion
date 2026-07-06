import pandas as pd
import nvector as nv

def main():
    df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/results_files/car_node_pers_person_1_location_results.csv')
    wgs84 = nv.FrameE(name='WGS84')

    for index, row in df.iterrows():
        car_gps = (row['Car_Latitude'], row['Car_Longitude'], row['Car_Altitude'])
        node_gps = (row['Node_Latitude'], row['Node_Longitude'], row['Node_Altitude'])
        obj_car_gps = (row['Car_Object_Latitude'], row['Car_Object_Longitude'], row['Car_Object_Altitude'])
        obj_node_gps = (row['Node_Object_Latitude'], row['Node_Object_Longitude'], row['Node_Object_Altitude'])

        point_car_gps = wgs84.GeoPoint(latitude=car_gps[0], longitude=car_gps[1], degrees=True)
        point_node_gps = wgs84.GeoPoint(latitude=node_gps[0], longitude=node_gps[1], degrees=True)
        point_obj_car_gps = wgs84.GeoPoint(latitude=obj_car_gps[0], longitude=obj_car_gps[1], degrees=True)
        point_obj_node_gps = wgs84.GeoPoint(latitude=obj_node_gps[0], longitude=obj_node_gps[1], degrees=True)

        # gt_gps = (row['GT_Latitude'], row['GT_Longitude'], row['GT_Altitude'])


        car_gps_to_node_gps_ECEF_vector = point_car_gps.to_ecef_vector() - point_node_gps.to_ecef_vector()
        car_gps_to_obj_car_gps_ECEF_vector = point_car_gps.to_ecef_vector() - point_obj_car_gps.to_ecef_vector()
        node_gps_to_obj_node_gps_ECEF_vector = point_node_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()
        obj_car_gps_to_obj_node_gps_ECEF_vector = point_obj_car_gps.to_ecef_vector() - point_obj_node_gps.to_ecef_vector()

        distance_euclidean_car_gps_to_node_gps = car_gps_to_node_gps_ECEF_vector.length
        distance_euclidean_car_gps_to_obj_car_gps = car_gps_to_obj_car_gps_ECEF_vector.length
        distance_euclidean_node_gps_to_obj_node_gps = node_gps_to_obj_node_gps_ECEF_vector.length
        distance_euclidean_obj_car_gps_to_obj_node_gps = obj_car_gps_to_obj_node_gps_ECEF_vector.length
        print(f"{row['Time']} {distance_euclidean_car_gps_to_node_gps} {distance_euclidean_car_gps_to_obj_car_gps} {distance_euclidean_node_gps_to_obj_node_gps} {distance_euclidean_obj_car_gps_to_obj_node_gps}")

if __name__ == '__main__':
    main()