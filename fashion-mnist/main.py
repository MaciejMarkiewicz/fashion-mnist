import knn
import conv_net
import dataset_augmenter
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# knn.run_training(train_images, train_labels, test_images, test_labels)
train_images, train_labels, test_images, test_labels = \
    dataset_augmenter.preprocess_data(train_images[:1000], train_labels[:1000], test_images[:100], test_labels[:100])
conv_net.run_training(train_images, train_labels, test_images, test_labels)
