# README #

We constructed three artificial neural network (NN) filters that can be used to enhance 3D images of neurites prior to automated tracing. Python codes for these filters are provided here along with three trained networks. 

## Overview ##
### Data ###

* The data folder contains two folders: L1 and Training_results. L1 folder contains 6 training images and their labels in tiff format for Neocortical Layer 1 Axons (http://diademchallenge.org/neocortical_layer_1_axons_readme.html) used in the Diadem Challenge (http://diademchallenge.org/challenge.html). Training_results is the default folder for saving the results of training. This folder also contains three trained networks.

### Codes ###

* The Python folder contains codes for training and testing the NN filters, and common_functions.py that includes functions used by these codes.

### Models ###

* We considered three network architectures (Figure 1): Shallow dense network, Multilayer dense network, and 3D U-Net. For each of these architectures, there is a code for training the network and for loading and testing the trained model. In addition, a network trained on Neocortical Layer 1 Axons is provided for each architecture.

### Input and output ###

* Input and output: Input and label images must be in an 8-bit multi-layer tiff format. The output is a numpy array of the same size as the input and the label. The output values are in the 0-1 range.

### Sample Data ###

* Sample data can be obtained at http://www.northeastern.edu/neurogeometry/resources/bouton-analyzer/

### User Manual and Demos ###

* User Manual is included in the repository
* Optimize Trace and Generate Profile GUI demo: https://www.youtube.com/watch?v=-QsEobWRVZE
* Detect and Track Boutons GUI demo: https://www.youtube.com/watch?v=UoGCRKXuuWc

### Contact ###

* Shih-Luen Wang wang.shihl@husky.neu.edu
* Armen Stepanyants a.stepanyants@northeastern.edu
* Python version 3.6, Tensorflow, and Keras 1.0 or later.
