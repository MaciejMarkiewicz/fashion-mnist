import knn
import conv_net
import dataset_augmenter
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# knn.run_training(train_images, train_labels, test_images, test_labels)

train_images, train_labels, val_images, val_labels, test_images, test_labels = \
    dataset_augmenter.preprocess_data(train_images[:100], train_labels[:100], test_images, test_labels)

# conv_net.run_training(train_images, train_labels, val_images, val_labels, test_images, test_labels)


# TEST:
# test_images = (test_images / 255).reshape(len(test_images), 28, 28, 1)
# test_labels = keras.utils.to_categorical(test_labels)
#
# model = keras.models.load_model('current_best.h5')
# model.summary()
# model.evaluate(test_images, test_labels, verbose=2)
