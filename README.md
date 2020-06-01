# Fashion-mnist ML models

This repository contains a couple of ML models trained for image recognition. 
The fashion-mnist dataset is used. 

# Methods

## Testing previous models

Choosing from the models implemented during previous classes I've decided to use
a KNN classifier, which should perform well on an image recognition task.
It can be found in the knn.py file. I had to modify the distance function so that it fits
new data - it's a simple Euclidean distance function, the weights are uniform. There's no image 
preprocessing/feature extraction apart from normalization, 
they are treated as 28x28 matrices and supplied to the distance function in this format.
This model is similar to the model from the benchmark site:

KNeighborsClassifier {"n_neighbors":5,"p":2,"weights":"uniform"}, accuracy: 0.851  

## Better model

For my go-to model I've chosen a convolutional neural network, which is also well suited for an image 
classification task. 

# Results

## KNN

I tested the k-value on a limited set of 10,000 train images and 2,000 validation images
and decided to go with k = 5 for the full-size test (actually half-size, because neither my laptop, nor Google Colab could handle it). 
The results obtained are as follows (accuracy):

10k train/2k validation:

- 0.793 for k=50

- 0.822 for k=9

- 0.823 for k=5

- 0.820 for k=3

- 0.808 for k=1

And for the 30k train/5k test:

- 0.839 for k=5

That corresponds to the official benchmark (0.851 accuracy when using a full set), so it seems to work just fine.

## CNN

THe results obtained for the network are better, than KNN. Training time is significantly shorter as well.
For the current version tests for the 10k train set tested on 2k test set give accuracy:

- 0.915 on the training set
- 0.893 on the test set

And for the full test:

- ___ on the training set
- ___ on the test set

# Usage 

Everything can be run by running the main.py file. Tensorflow/Kreas, numpy and matplotlib libraries
are needed. I used Python 3.7 Anaconda distribution. The fashion-mnist set is included with tensorflow, 
so no additional downloading is necessary. KNN model can't be downloaded, 
as it requires specific test data. CNN model is available for download form the fashion_mnist_conv_net folder,
in a SavedModel Keras format, and can be used easily in Keras as any other saved model.