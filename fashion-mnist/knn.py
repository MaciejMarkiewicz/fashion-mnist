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


def knn_model_selection(X_val, X_train, y_val, y_train, k_values):
    labels = sort_train_labels_knn(distance_function(X_val, X_train), y_train)
    prediction = [p_y_x_knn(labels, k) for k in k_values]
    errors = [classification_error(pred, y_val) for pred in prediction]
    min_error_index = np.argmin(errors)
    return dict(zip(k_values, errors)), k_values[min_error_index]


def knn_test(X_train, X_test, y_train, y_test, k):
    labels = sort_train_labels_knn(distance_function(X_test, X_train), y_train)
    return classification_error(p_y_x_knn(labels, k), y_test)


def knn_on_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_set = train_images[:1000, :]
    train_set_labels = train_labels[:1000]
    val_set = train_images[1000:1200, :]
    val_set_labels = train_labels[1000:1200]
    test_set = test_images[:200, :]
    test_set_labels = test_labels[:200]

    models, best_k = knn_model_selection(val_set, train_set, val_set_labels, train_set_labels, [1, 5, 9])

    print(models)
    print(knn_test(train_set, test_set, train_set_labels, test_set_labels, best_k))


# res 10/2k for val set
# {1: 0.1845, 3: 0.17, 5: 0.1615, 9: 0.1685, 20: 0.183, 50: 0.2065}
# res for test set 0.1775

# TODO update readme!
