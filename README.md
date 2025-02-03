\title{ADAS PROJECT}
In our project we focused on the creation of a neural network with the aim of reconstructing the roadway starting from the position of the blue and yellow cones
in world coordinates that delimit its boundaries.

**Initial Input**: Some files containing the (x, y) coordinates of yellow and blue cones collected from a given frame.

**Goal**: Reconstruction of roadway boundaries and the centerline

*Project Development Pipeline*

_Dataset Preprocessing_: 
- feature selection to remove unclear or erroneous frames and applying transformations to standardize the data.
- data augmentation step was then conducted to enrich the dataset and enhance the model's robustness.
- convertion into a grid format to serve as input for the neural network

_Neural Network Model Development_
Various neural network models were developed and tested to identify the most suitable architecture for solving the 
specific problem.

_Performance Evaluation and Results Visualization_
This step aimed to evaluate the model's effectiveness in reconstructing the road and localizing the cones accurately.

