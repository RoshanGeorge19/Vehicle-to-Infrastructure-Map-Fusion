clear; clc; close all;
%% SLAM Edge Map V2 Rotate 15deg.
pointCloudData_edge_map = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\CAR\Scenario-2\' ...
    'slam\v2\edge_map_only_v2\edge_map_v2.pcd']); 

% Extract (x, y, z) coordinates from the point cloud data
points = pointCloudData_edge_map.Location;

% Define rotation angle in radians
theta_lidar_to_local = deg2rad(-15);

% Define the rotation matrix for rotation about the Z-axis
R_LidarToLocal = [cos(theta_lidar_to_local), -sin(theta_lidar_to_local), 0;
                  sin(theta_lidar_to_local),  cos(theta_lidar_to_local), 0;
                  0,                          0,                         1];

% Apply the rotation matrix to the point cloud
rotatedPoints = (R_LidarToLocal * points')'; % Transpose, apply rotation, and transpose back

% Update the point cloud with the rotated points
edge_map_rotated = pointCloud(rotatedPoints);
%% Car VLP-16 Rotate 15deg.
pointCloudData = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\CAR\Scenario-3\' ...
    'point_cloud_data\pcd_out\4076.540-Car_Scene-3.pcd']); 

points = pointCloudData.Location;

% Get the dimensions of the points data
[rows, cols, dimension] = size(points);  % Instead of Nx3, we have rows x cols x 3

% Check if the 3rd dimension equals 3 (XYZ coordinates)
if dimension ~= 3
   error('The point cloud does not have 3D coordinates in the third dimension.');
end

% Reshape the points data from (rows x cols x 3) to (N x 3, where N = rows*cols)
pointsFlat = reshape(points, [], 3);  % This flattens the (rows x cols x 3) to (rows*cols) x 3

% Define rotation angle in radians
theta_lidar_to_local = deg2rad(-15);

% Define the rotation matrix for rotation about the Z-axis
R_LidarToLocal = [cos(theta_lidar_to_local), -sin(theta_lidar_to_local), 0;
                  sin(theta_lidar_to_local),  cos(theta_lidar_to_local), 0;
                  0,                           0,                        1];

% Apply the rotation matrix to the flattened points (N x 3 matrix)
rotatedFlatPoints = (R_LidarToLocal * pointsFlat')';  % Rotating (N x 3) -> Transpose, multiply, transpose

% Reshape the rotated points back into the original structure (rows x cols x 3)
rotatedPoints = reshape(rotatedFlatPoints, rows, cols, 3);

% Create a new pointCloud object with the rotated points
instant_pcd_rotated = pointCloud(rotatedPoints);
%% Node VLP-16
pointCloudData_2 = pcread(['G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\Cepton\Scenario3-2\' ...
    'point_cloud_data\node_velo_pcd_out\4079.770-Cepton_Velo_Scene-3-2.pcd']); 
%% Node VLP-16 Edit 2
pointCloudData = pcread(['C:\Users\Roshan George\Desktop\PCD_Edge_Instant_Edit\' ...
    'From Workspace_instant_velopcd_Edit\velo_instant_edit_2.pcd']); %#ok<NASGU>
%%
% Define the translation vector [tx, ty, tz] for 20 meters along the Y-axis
translationVector = [0, 20, 0];  % No translation on X/Z axis, 20m on Y-axis
    
% Create a rigidtform3d object with no rotation and the translation vector [0, 20, 0]
rigidTransform = rigidtform3d(eye(3), translationVector);  % Identity rotation matrix + translation

% Display the rigid transformation matrix
% disp('3D Rigid Transformation Matrix (Translation Only on Y-axis):');
% disp(rigidTransform.A);  % The A property returns the 4x4 transformation matrix
%%
% Visualize the original and rotated point cloud for comparison
figure;
subplot(1, 2, 1);
pcshow(pointCloudData);
title('Original Point Cloud');

subplot(1, 2, 2);
pcshow(rotatedPointCloudData);
title('Rotated Point Cloud');
