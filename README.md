# README #

We constructed three artificial neural network (NN) filters that can be used to enhance 3D images of neurites prior to automated tracing. Python codes for these filters are provided here along with three trained networks. 

### Requirements ###

* Python version 3.6, Tensorflow, and Keras 1.0 or later.

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
