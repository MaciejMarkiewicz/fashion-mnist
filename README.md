# Fashion-mnist ML models

This repository contains a couple of ML models trained for image recognition. 
The fashion-mnist dataset is used. 

# Methods

Choosing from the models implemented during previous classes I've decided to use
a KNN classifier, which should perform well on an image recognition task.
It can be found in the knn.py file. I had to modify the distance function so that it fits
new data - it's a simple Euclidean distance function, the weights are uniform. There's no image preprocessing, 
they are treated as 28^2 vectors and supplied to the distance function in this format.
The model is similar to the model from the benchmark site:

KNeighborsClassifier {"n_neighbors":5,"p":2,"weights":"uniform"}, accuracy: 0.851  

# Results

KNN

I played a little with the k-value on a limited set of 10,000 train images and 2,000 test images
and decided to go with k = 5 for the full-size test. The results obtained are as follows (error rates):

10k/2k:

0.207 for k=50

0.178 for k=9

0.1775 for k=5

0.1805 for k=3

0.1925 for k=1

And for the full size test:

.

That corresponds to the official benchmark, so it seems to work just fine.

# Usage 

Everything can be run by running the main.py file. Tensorflow/Kreas, numpy and matplotlib libraries
are needed. I used Python 3.7 Anaconda distribution. The fashion-mnist set is included with tensorflow, 
so no additional downloading is necessary, as it is automatic. KNN model can't be downloaded, 
as it requires specific test data.