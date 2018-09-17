% the Shallow and Deep NB-SRCNN for n layers + one concatente layer:
% NB-SRCNN is for Non-blind deblurring super-resolution CNN

function im_h_y = NB_SRCNN_Concat(im_l_y, model, scale)

weight = model.weight;
bias = model.bias;
im_y = single(imresize(im_l_y,scale,'bicubic'));

%% the first layer
convfea1 = vl_nnconv(im_y,weight{1},bias{1}, 'Pad',4);
convfea1 = vl_nnrelu(convfea1);

%% the second layer
convfea2 = vl_nnconv(convfea1,weight{2},bias{2}, 'Pad', 2);
convfea2 = vl_nnrelu(convfea2);

%% concatenated layer
convfea12 = vl_nnconcat({convfea1, convfea2});

%% mapping layer
convfea3 = vl_nnconv(convfea12,weight{3},bias{3}, 'Pad', 2);
convfea3 = vl_nnrelu(convfea3);

%% for 4 layers + one concatente layer:
% convfea4 = vl_nnconv(convfea3,weight{4},bias{4}, 'Pad', 2);  % Shallow Net (4 layers)
% im_h_y = convfea4;

%% for 5 layers + one concatente layer:
convfea4 = vl_nnconv(convfea3,weight{4},bias{4}, 'Pad', 2);
convfea4 = vl_nnrelu(convfea4);
convfea5 = vl_nnconv(convfea4,weight{5},bias{5}, 'Pad', 2);  % Deeper Net ( 5 layers )
im_h_y = convfea5;
