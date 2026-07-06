clear, clc, close all;
edge_labels = load("id-307_id-356_edge_map_labels.mat");
object_labels= load("id-307_id-356_static_sign_2_labels.mat");
gTruth = object_labels.gTruth;
labelData = gTruth.LabelData;
dim = edge_labels.gTruth.LabelData.static_sign_2(1,4:6);
%%
for ii=1:numel(gTruth.LabelData)
    if ~isempty(labelData)
       labelData.static_sign_2(ii,4:6) = dim;
    end
end
%%
newGTruth = groundTruthLidar(gTruth.DataSource, gTruth.LabelDefinitions, labelData);