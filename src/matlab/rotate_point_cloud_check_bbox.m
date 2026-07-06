clear; clc; close all;
point_cloud = readPointCloud(lasFileReader(['G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\Cepton\Scenario3-2\car_cepton_overlap_1\person_1' ...
    '\out\4153.023-Cepton_Scene-3-2.las'])); 

points = point_cloud.Location;
rotation = deg2rad(15);
z_rotation = [cos(rotation), -sin(rotation), 0;
                  sin(rotation),  cos(rotation), 0;
                  0,                          0,                         1];
rotatedPoints = (z_rotation * points')'; % Transpose, apply rotation, and transpose back
point_cloud_rotated = pointCloud(rotatedPoints);


%%
% Visualize the original and rotated point cloud for comparison
figure;
subplot(1, 2, 1);
pcshow(point_cloud);
title('Original Point Cloud');

subplot(1, 2, 2);
pcshow(point_cloud_rotated);
title('Rotated Point Cloud');
