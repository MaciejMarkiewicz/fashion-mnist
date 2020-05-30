import numpy as np
from tensorflow import keras

NUMBER_OF_CLASSES = 10
fashion_mnist = keras.datasets.fashion_mnist


def distance_function(X, X_train):
    return np.array([[np.linalg.norm(test_o-train_o) for train_o in X_train] for test_o in X])


def sort_train_labels_knn(Dist, y):
    return y[np.argsort(Dist)]


def p_y_x_knn(y, k):
    return np.array([np.bincount(row[:k], minlength=NUMBER_OF_CLASSES) for row in y]) / k


def classification_error(p_y_x, y_true):
    y_pred = [np.argmax(row) for row in p_y_x]
    return sum(y_pred[i] != y_true[i] for i in range(y_true.shape[0])) / len(y_true)


def knn_training(X_val, X_train, y_val, y_train, k_values):
    labels = sort_train_labels_knn(distance_function(X_val, X_train), y_train)
    probabilities = [p_y_x_knn(labels, k) for k in k_values]
    errors = [classification_error(prob, y_val) for prob in probabilities]
    return dict(zip(k_values, errors))


def knn_on_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    np.reshape(train_images, (60000, 28*28))
    np.reshape(test_images, (10000, 28*28))

    # train_images = train_images[:10000, :]
    # train_labels = train_labels[:10000]
    # test_images = test_images[:2000, :]
    # test_labels = test_labels[:2000]

    res = knn_training(test_images, train_images, test_labels, train_labels, [5])

    print(res)
