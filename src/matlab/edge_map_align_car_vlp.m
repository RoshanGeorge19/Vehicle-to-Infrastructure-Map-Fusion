clear; clc; close all

movingPtCloud = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data' ...
    '\lidar_data\CAR\Scenario-3\point_cloud_edited_alignment_node_car\car_velo_instant_rotated_car_local_cs.pcd']);
fixedPtCloud = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data' ...
    '\lidar_data\CAR\Scenario-3\point_cloud_edited_alignment_node_car\edge_map_rotated_car_local_cs.pcd']);

translationVector = [0, 20, 0]; 
initialTform = rigidtform3d(eye(3), translationVector);  % Identity rotation matrix + translation

% Register the point clouds.
tform = pcregistericp(movingPtCloud,fixedPtCloud,Metric="planeToPlane", ...
    InlierRatio=1.000000,MaxIterations=3000,Tolerance=[0.010000 0.500000],InitialTransform=initialTform);

disp(tform.Translation)