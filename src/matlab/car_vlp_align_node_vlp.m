clear; clc; close all

fixedPtCloud = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data' ...
    '\lidar_data\CAR\Scenario-3\point_cloud_edited_alignment_node_car\car_velo_instant_rotated_car_local_cs.pcd']);
movingPtCloud = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data' ...
    '\lidar_data\CAR\Scenario-3\point_cloud_edited_alignment_node_car\node_velo_edit_3_ car_persons_removed.pcd']);

% Remove the ground from the point clouds.
gridResolution = 1.000000;
[~,movingPtCloud] = segmentGroundSMRF(movingPtCloud,gridResolution,MaxWindowRadius=18.000000,SlopeThreshold=0.150000,ElevationThreshold=0.500000,ElevationScale=1.250000);
[~,fixedPtCloud] = segmentGroundSMRF(fixedPtCloud,gridResolution,MaxWindowRadius=18.000000,SlopeThreshold=0.150000,ElevationThreshold=0.500000,ElevationScale=1.250000);

% Register the point clouds.
tform = pcregistericp(movingPtCloud,fixedPtCloud,Metric="planeToPlane",InlierRatio=1.000000,MaxIterations=3000,Tolerance=[0.010000 0.500000]);

disp(tform.Translation)