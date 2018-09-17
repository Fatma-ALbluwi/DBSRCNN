clear all;
close all;
%% settings
folder = 'Test/Set5';
savepath = 'test_mscale_blur1.h5';
size_input = 33;
size_label = 33;
multi_scale = [3];
stride = 21;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    if size(image, 3) > 1
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));
    else
        image = im2double(image);
    end
    for k = 1 : length(multi_scale)
        scale = multi_scale(k);
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input=imgaussfilt(im_label, 1);  % to blur images here blur =1, you can change it to equal=1,2,3,or 4
    im_input = imresize(imresize(im_input,1/scale,'bicubic'),[hei,wid],'bicubic'); % to resize images using factor = 3

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 2;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
