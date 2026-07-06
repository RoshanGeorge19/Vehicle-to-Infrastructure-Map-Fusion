addpath 'G:\Documents\MATLAB\sort'

LabelData = readtable('Cepton\Scenario2-1\extractedLabelData_Cepton.csv');
timeData = readtable('Cepton\Scenario2-1\timeForLables_259-308.csv');

% Find unique values of time in LabelData
uniqueTimes = unique(LabelData.Time);

% Sort the unique time values naturally
sortedUniqueTimes = natsort(uniqueTimes);

% Create a lookup table
lookupTable = table(sortedUniqueTimes, num2cell(timeData.Time), 'VariableNames', {'Time', 'Data'});

% Iterate over each unique time value in the lookup table
for i = 1:numel(lookupTable.Time)
    % Find the row indices in LabelData corresponding to the current time value
    rowsToUpdate = strcmpi(LabelData.Time, lookupTable.Time{i});
    
    % Extract the corresponding value from the lookup table data
    cellData = lookupTable.Data{i};
    
    % Convert numeric data to strings
    cellData = arrayfun(@num2str, cellData, 'UniformOutput', false);
    
    % Replace the values in LabelData.Time with the corresponding values from the lookup table
    LabelData.Time(rowsToUpdate) = cellData(1);
end

writetable(LabelData, 'Cepton\Scenario2-1\extractedLabelData_Cepton.csv');
