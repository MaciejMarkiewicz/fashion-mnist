import knn
import conv_net
import dataset_augmenter
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# knn.run_training(train_images, train_labels, test_images, test_labels)

# train_images, train_labels, val_images, val_labels = \
#     dataset_augmenter.preprocess_data(train_images, train_labels)
#
# conv_net.run_training(train_images, train_labels, val_images, val_labels)


# TEST:
test_images = (test_images / 255).reshape(len(test_images), 28, 28, 1)
test_labels = keras.utils.to_categorical(test_labels)

model = keras.models.load_model('trained_model/model.h5')
model.summary()
model.evaluate(test_images, test_labels, verbose=2)
