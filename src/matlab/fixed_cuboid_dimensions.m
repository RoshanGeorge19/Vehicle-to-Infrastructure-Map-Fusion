%% Workflow
% Load ground_removed into lidarLabeller.
% Make one label - person_1.
% Export label defintions to workspace as gTruth_test.
% Append gTruth_test.DataSource to gTruthFixedDim.
% clear; clc; close all;
%% Node
% load('id-720_id-869_person_1_labels.mat')
% fixed_length = 0.6;
% fixed_width = 0.5;
% fixed_height = 1.7;
% fixedCuboidDimensions = [fixed_length, fixed_width, fixed_height]; 
% labelData = gTruth.LabelData;
% newLabelData = labelData; 
% for k = 1:numel(labelData)
%     newLabelData{k,:}(4:6) = fixedCuboidDimensions;
% end
% gTruthFixedDim = groundTruthLidar(gTruth_test.DataSource, gTruth.LabelDefinitions, newLabelData);
%% Car
load('id-755_id-904_person_1_labels.mat')
fixed_length = 0.6;
fixed_width = 0.5;
fixed_height = 1.7;
fixedCuboidDimensions = [fixed_length, fixed_width, fixed_height]; 
labelData = gTruth.LabelData;
newLabelData = labelData; 
for k = 1:numel(labelData)
    if ~isempty(newLabelData{k,:}{:})
        newLabelData{k,:}{:}(4:6) = fixedCuboidDimensions;
    else
        newLabelData{k,:}{:} = [0 0 0 0 0 0 0 0 0];
    end
end
gTruthFixedDim = groundTruthLidar(gTruth_test.DataSource, gTruth.LabelDefinitions, newLabelData);
