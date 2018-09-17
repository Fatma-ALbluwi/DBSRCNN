# DBSRCNN
Deblurring Super-Resolution Convolutional Neural Network

# DBSRCNN-Keras

This code is to process the blurred low-resolution images to get deblurred high-residual images.
If this code is helpful for you, please cite this paper: "IMAGE DEBLURRING AND SUPER-RESOLUTION USING DEEP CONVOLUTIONAL NEURAL NETWORKS".

## Dependencies

1. TensorFlow.
2. Keras.
3. Matconvnet.  

## Generating data

1. blur images by gaussian filter (imgaussfilt) at different levels (sigma = 1, 2, 3 and 4).
2. resize images with 'bicubic' function using upscaling factor = 3, published papers recently generally use Matlab to produce low-resolution image.
3. For a fair comparison with SRCNN network; training set Yang91 is used.

## Training

1. generate training patches using matlab: run generate_train.m and generate_test.m.
2. use Keras with TensorFlow (tf) as a backend to train DBSRCNN model; Adam is used to optimizing the network for fast convergence: run DBSRCNN_train.py to produce DBSRCNN_blur model. 
3. convert Keras model to .Mat for testing using Matconvnet: run load_save.py first, then run save_model.m to produce Matconvnet model. 
4. run NB_SRCNN_Concat_blur_test.m in “test” folder to test the model; Set5 and Set14 are used as testing data.


##  Note
This code is based on Keras-2.