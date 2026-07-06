clear; clc; close all
%% Load ground truth labels
load('G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\Cepton\Scenario3-2\car_cepton_overlap_1\person_1\out\session_labels_csv\id-720_id-869_fixed_person_1_labels.mat')

% Define the path of the LiDAR dataset
lidarDataFolder = 'G:\Documents\Pycharm Projects\Work_Package_1\data\lidar_data\Cepton\Scenario3-2\car_cepton_overlap_1\person_1\out';

% Load groundTruthLidar object
gtLidar = gTruth;  % Ensure `groundTruthLidar` is loaded

% Get label definitions and label data
labelNames = gtLidar.LabelDefinitions.Name;
labelData = gtLidar.LabelData;  % This is typically a table

% Find all LAS files
lidarFiles = dir(fullfile(lidarDataFolder, '*.las'));
lidarFiles = natsortfiles({lidarFiles.name});  % Requires `natsortfiles.m`

% Check LAS file and label consistency
numFrames = min(length(lidarFiles), height(labelData));
if length(lidarFiles) ~= height(labelData)
    warning('Number of LAS files (%d) does not match labelData rows (%d)!', length(lidarFiles), height(labelData));
end

%% 🌟 **Loop Over All Frames**
for frameIndex = 1:5
    lasFilePath = fullfile(lidarDataFolder, lidarFiles{frameIndex});
    
    % Read the LAS file
    lasReader = lasFileReader(lasFilePath);
    ptCloud = readPointCloud(lasReader); 

    % Extract bounding box info
    bbox_found = false;
    for i = 1:numel(labelNames)
        labelEntries = labelData.(labelNames{i});

        % Check if labelEntries is a cell array or direct matrix
        if iscell(labelEntries)
            objectAnnotations = labelEntries{frameIndex}; 
        else
            objectAnnotations = labelEntries(frameIndex, :); 
        end

        if ~isempty(objectAnnotations) && ~all(isnan(objectAnnotations(:)))
            bbox = objectAnnotations(1, :); 
            bbox_found = true;
            break;
        end
    end

    if ~bbox_found
        warning('No bounding box found in frame %d. Skipping...', frameIndex);
        continue; % Skip this frame and move to the next
    end

    % Extract the bounding box center and dimensions
    center = bbox(1:3);
    dims = bbox(4:6);

    % Define zoom boundaries
    zoomRange = 2 * norm(dims);  
    xLimits = [center(1) - zoomRange, center(1) + zoomRange];
    yLimits = [center(2) - zoomRange, center(2) + zoomRange];
    zLimits = [center(3) - zoomRange, center(3) + zoomRange];

    % 🌟 **Create Subplots for Different Views**
    figure;

    % --- **Subplot 1: XY View (Top-Down View)**
    subplot(1,3,1);
    pcshow(ptCloud);
    hold on;
    drawLidarBoundingBox3D(bbox, 'r', 2);
    view(0, 90);
    xlim(xLimits);
    ylim(yLimits);
    title(sprintf('Frame %d - XY View', frameIndex));
    xlabel('X'); ylabel('Y');

    % --- **Subplot 2: YZ View (Side View)**
    subplot(1,3,2);
    pcshow(ptCloud);
    hold on;
    drawLidarBoundingBox3D(bbox, 'r', 2);
    view(0, 0);
    ylim(yLimits);
    zlim(zLimits);
    title(sprintf('Frame %d - YZ View', frameIndex));
    xlabel('Y'); ylabel('Z');

    % --- **Subplot 3: XZ View (Front View)**
    subplot(1,3,3);
    pcshow(ptCloud);
    hold on;
    drawLidarBoundingBox3D(bbox, 'r', 2);
    view(90, 0);
    xlim(xLimits);
    zlim(zLimits);
    title(sprintf('Frame %d - XZ View', frameIndex));
    xlabel('X'); ylabel('Z');

    hold off;

    % 📸 **Optional: Save Each Frame as an Image**
    % saveas(gcf, sprintf('frame_%d.png', frameIndex));

    % ⏸ Pause Between Frames for Visualization
    pause(0.01);  % Adjust delay as needed
end

%% FUNCTION: Draw 3D Bounding Box
function drawLidarBoundingBox3D(box, color, lineWidth)
    center = box(1:3);
    dims = box(4:6);
    angles = box(7:9);

    % Generate corner coordinates of cuboid
    bboxCorners = generate3DBoxCorners(center, dims, angles);

    % Define edges of the box
    edges = [1,2; 2,3; 3,4; 4,1;  
             5,6; 6,7; 7,8; 8,5;  
             1,5; 2,6; 3,7; 4,8]; 

    % Plot the cuboid
    for k = 1:size(edges,1)
        plot3(bboxCorners(edges(k,:),1), bboxCorners(edges(k,:),2), bboxCorners(edges(k,:),3), ...
              'Color', color, 'LineWidth', lineWidth);
    end
end

%% FUNCTION: Generate 3D Bounding Box Corners
function corners = generate3DBoxCorners(center, dims, angles)
    l = dims(1) / 2;
    w = dims(2) / 2;
    h = dims(3) / 2;

    boxCorners = [-l, -w, -h;
                  l, -w, -h;
                  l,  w, -h;
                 -l,  w, -h;
                 -l, -w,  h;
                  l, -w,  h;
                  l,  w,  h;
                 -l,  w,  h];

    % Create rotation matrices
    yaw = angles(3); pitch = angles(2); roll = angles(1);

    Ryaw = [ cosd(yaw), -sind(yaw), 0;
             sind(yaw),  cosd(yaw), 0;
             0,         0,         1];

    Rpitch = [ cosd(pitch), 0, sind(pitch);
               0,         1, 0;
              -sind(pitch), 0, cosd(pitch)];

    Rroll = [ 1, 0, 0;
              0, cosd(roll), -sind(roll);
              0, sind(roll), cosd(roll)];

    % Apply full rotation transformation
    R = Ryaw * Rpitch * Rroll;
    rotatedCorners = (R * boxCorners')';

    % Translate the corners to final position
    corners = rotatedCorners + center;
end