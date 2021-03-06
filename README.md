
# DBSRCNN
Deblurring Super-Resolution Convolutional Neural Network.

# DBSRCNN Network 

![dbsrcnn arct](https://user-images.githubusercontent.com/16929158/45629859-4bd2dc80-ba8f-11e8-82f4-409c28a32777.png)

# DBSRCNN-Keras

This code is to process the blurred low-resolution images to get deblurred high-residual images.

If this code is helpful for you, please cite this paper: Image Deblurring And Super-Resolution Using Deep Convolutional Neural Networks,
F. Albluwi, V. Krylov and R. Dahyot, IEEE International Workshop on Machine Learning for Signal Processing (MLSP 2018 <http://mlsp2018.conwiz.dk/home.htm> ), September 2018, Aalborg, Danemark.

## Dependencies

1. Python 3.6.5
2. TensorFlow 1.1.0.
3. Keras 2.2.2.
4. Matlab.
5. Matconvnet.  

## Generating data

1. blur images by gaussian filter (imgaussfilt) at different levels (sigma = 1, 2, 3 and 4).
2. resize images with 'bicubic' function using upscaling factor = 3, published papers recently generally use Matlab to produce low-resolution image.
3. For a fair comparison with SRCNN network; training set Yang91 is used.

## Training

1. generate training patches using matlab: run generate_train.m and generate_test.m.
2. use Keras with TensorFlow (tf) as a backend to train DBSRCNN model; Adam is used to optimizing the network for fast convergence: run DBSRCNN_train.py to produce DBSRCNN_blur model. 
3. convert Keras model to .Mat for testing using Matconvnet: run load_save.py first, then run save_model.m to produce Matconvnet model. 
4. run NB_SRCNN_Concat_blur_test.m in “test” folder to test the model; Set5 and Set14 are used as testing data.

## Some Qualitative Results

![img1](https://user-images.githubusercontent.com/16929158/46291571-c408ca00-c586-11e8-8c42-28ca32f50a6f.png)
![img6](https://user-images.githubusercontent.com/16929158/46292371-af2d3600-c588-11e8-892f-d1bf08a1085b.png)
![img5](https://user-images.githubusercontent.com/16929158/46292284-670e1380-c588-11e8-8c68-acc9844df88c.png)
