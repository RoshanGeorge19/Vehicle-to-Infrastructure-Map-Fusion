clear; clc; close all;
%%
load(['G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\Cepton\Scenario3-2\car_cepton_overlap_1\person_1\out\session_labels_csv\' ...
    'id-720_id-869_fixed_person_1_labels.mat'])
%%
allLabels = {};
numFrames = height(gTruth.LabelData);
%%
for i = 1:numFrames
    currentLabels = gTruth.LabelData(i, :);
    timeStamp = currentLabels.Time;
    object_name = 'person_1';
    object_labels = gTruth.LabelData.person_1(i,:);

    if ~isempty(object_labels)
        num_object_labels = size(object_labels, 1);
        for j = 1:num_object_labels
            if isempty(object_labels(1:3))
                object_labels(1,:) = [0 0 0 0 0 0 0 0 0];
            end

            labelData = object_labels(1,:);
            labelPosition = labelData(1:3);
            labelDimensions = labelData(4:6);
            labelRotation = labelData(7:9);
            labelStruct = struct( ...
                'Time', timeStamp, ...
                'Name', object_name, ...
                'Position', labelPosition, ...
                'Dimensions', labelDimensions, ...
                'Rotation', labelRotation ...
            );
            allLabels{end+1} = labelStruct;
        end
    end
end
%%
allLabels = [allLabels{:}];
disp(allLabels);
labelTable = struct2table(allLabels);
writetable(labelTable, 'session_converted_csv.csv');
