close all;
clear all; 

%%
run matconvnet\matlab\vl_setupnn;

addpath('utils')
load('models\DBSRCNN_blur1.mat')

% set parameters
up_scale = 3;

%% change the number according to sigma (1,2,3,4) of non-blind blurring in training.
 j=1; % values 1,2,3,4 (according to blurring level)

% to save the result as a matrix 
mkdir('Result');

%% Data for testing gray or colourful images
% for colourful images:  'Data\Set5\' or 'Data\Set14\'
im_path = 'Data\Set5\';  

% for gray images: 'Data_gray\Set5_gray\' or 'Data_gray\Set14_gray\'
%im_path = 'Data_gray\Set5_gray\'; 

im_dir = dir( fullfile(im_path, '*bmp') );
im_num = length( im_dir );

%% save scores; this is for non-blind blurring
scores(im_num,4)=0;

%%
for img = 1:im_num
X = imread( fullfile(im_path, im_dir(img).name) );
grd = X;
if size(X,3) == 3
    X = rgb2ycbcr(X);
    X = double(X(:,:, 1));
else
    X = double(X);
end
X = modcrop(X, up_scale);
grd = modcrop(grd, up_scale);
X = double(X);
[row, col, ~] = size(X);

%% Generate blurred LR image.
 
im_l = imgaussfilt(X, j);  % to blur images
im_l = imresize(im_l, 1/up_scale, 'bicubic')/255; % downsampling the images by using factor = up_scale

%% for blur0/LR

% im_l = imresize(X, 1/up_scale, 'bicubic')/255;

%% DBSRCNN Network

im_h_y = NB_SRCNN_Concat(im_l, model, up_scale);
im_h = double(im_h_y * 255);

%% Show

lr = imresize(grd, 1/up_scale, 'bicubic');

% for colourful images:
if size(lr, 3) == 3
    lr = rgb2ycbcr(lr);
    xy = uint8(im_l*255);
    xcb = lr(:, :, 2);
    xcr = lr(:, :, 3);
    lr(:,:, 1) = xy;
    % only the first channel is changed but the rest of channels is the same
    bic(:, :, 1) = imresize(xy, up_scale, 'bicubic');
    bic(:, :, 2) = imresize(xcb, up_scale, 'bicubic');
    bic(:, :, 3) = imresize(xcr, up_scale, 'bicubic');
    %im_bic = low resolution 
    im_bic = ycbcr2rgb(bic);
    bic(:, :, 1) = uint8(im_h);
    our = ycbcr2rgb(bic);
    lr = ycbcr2rgb(lr);
else
    % for gray images:
    %im_bic = imresize(lr, up_scale, 'bicubic');
    im_bic = imresize(im_l, up_scale, 'bicubic')*255;
    %im_bic = uint8(im_l*255);
    our = uint8(im_h);
end
clear bic;

%im_bic = imgaussfilt(im_bic, 1);

grd = shave(grd, [up_scale, up_scale]);
our = shave(our, [up_scale, up_scale]);
im_bic = shave(im_bic, [up_scale, up_scale]);
%% 

% lr=low resolution, grd = high resolution, im_bic = image bicubic, our = DBSRCNN 
savefile( lr, grd, our, im_bic, im_dir(img).name);

%% Evaluation
X = shave(uint8(X), [up_scale, up_scale]);
im_h = shave(uint8(im_h), [up_scale, up_scale]);
pp_psnr = compute_rmse(X, im_h);
pp_psnr_bic = compute_rmse(X, im_bic);

scores(img, 1) = pp_psnr;
scores(img, 2) = pp_psnr_bic;
scores(img, 3) = ssim(X, im_h);
%scores(img, 4) = ssim(X, im_bic);
scores(img, 5) = j;
end

scores(im_num+1,:)= mean(scores);
save Result\scores scores;