# README #

We constructed three artificial neural network (NN) filters that can be used to enhance 3D images of neurites prior to automated tracing. Python codes for these filters are provided here along with three trained networks. 

## Overview ##

### Data ###

* The data folder contains two subfolders: L1 and Training_results. L1 subfolder contains 6 training images and their labels in tiff format for Neocortical Layer 1 Axons (http://diademchallenge.org/neocortical_layer_1_axons_readme.html) used in the Diadem Challenge (http://diademchallenge.org/challenge.html). Training_results is the default folder for saving the results of training. This folder also contains three trained networks.

### Codes ###

* The Python folder contains codes for training and applying the NN filters, and common_functions.py that includes functions used by these codes.

### Models ###

* We considered three network architectures (https://github.com/neurogeometry/NNfilters/blob/master/Image%20enhancement.pdf): Shallow dense network, Multilayer dense network, and 3D U-Net. For each of these architectures, there is a code for training the network and for loading and applying the trained model. In addition, a network trained on Neocortical Layer 1 Axons is provided for each architecture.

## How to use ##

### Requirements ###

* Python version 3.6, Tensorflow, and Keras 1.0 or later.

### Input and output ###

* Input and output: Input and label images must be in an 8-bit multipage tiff format. The output is a numpy array of the same size as the input and the label. The output values are in the 0-1 range.

### Training NN filters ###

* The user must specify the paths to the images and labels used for training and validation, learning rate, dropout ratio, batch size, maximum number of training steps, and plotting frequency. Input and output sizes can also be adjusted by the user. For Shallow and Multilayer networks the training loss is visualized in Tensorboard (https://www.tensorflow.org/guide/summaries_and_tensorboard). For 3D U-Net the loss is displayed with pyplot (https://matplotlib.org/api/pyplot_api.html).

### Applying NN filters ###

* To filter an image, the user must provide the paths to the image and the model. Maximum intensity projection of the filtered image will be displayed along with those for the original image and the label (if any). 

### Contact ###

* Shih-Luen Wang wang.shihl@husky.neu.edu
* Armen Stepanyants a.stepanyants@northeastern.edu

