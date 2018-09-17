%% save model / .mat
clc;
clear all;
close all;
model = {};

for k = 1 : 5
    strw = ['w',num2str(k-1),'.mat'];
    load(strw)
    model.weight{k} = array;
    strb = ['b',num2str(k-1),'.mat'];
    load(strb)
    model.bias{k} = array;
end
save DBSRCNN_blur1 model;  % to save model blur1, change it to DBSRCNN_blur2/ DBSRCNN_blur3 / or DBSRCNN_blur4
