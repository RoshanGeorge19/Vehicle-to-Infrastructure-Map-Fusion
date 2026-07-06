clear; clc; close all;
%%
load('G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\Cepton\Scenario3-2\car_cepton_overlap_1\person_1\out\session_labels_csv\id-720_id-869_person_1_labels_2.mat');

% Step 1: Extract existing label data into a variable
oldLabelData = gTruth.LabelData;

% Step 2: Modify or edit the label data as needed
% (In this case, copying data from the first row to the entire column)
newLabelData = oldLabelData; 
newLabelData{:,:}(:,1) = newLabelData{:,:}(:,1) - 1;
newLabelData{:,:}(:,3) = newLabelData{:,:}(:,1) - 1.5;

% newLabelData{:,1} = oldLabelData{1,1}; % Modify as you need
% 
% % Step 3: Create a new groundTruthLidar object with the modified LabelData
% % Assuming `gTruth` has other properties like ROIMetadata and PointCloudLabelDefinitions you need to keep
gTruthNew = groundTruthLidar(gTruth.DataSource, gTruth.LabelDefinitions, newLabelData);
% 
% % Now `gTruthNew` has the updated LabelData.