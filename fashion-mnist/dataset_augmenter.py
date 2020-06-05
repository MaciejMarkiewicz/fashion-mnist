import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from tensorflow import keras
import random

NUMBER_OF_AUGMENTATIONS = 3
TRAIN_SET_SIZE = 50000


def elastic_transform(image, alpha_range, sigma):
    random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def data_generator():
    return keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0,
        height_shift_range=0,
        zoom_range=0.0,
        horizontal_flip=True,
        preprocessing_function=lambda image: elastic_transform(image, alpha_range=8, sigma=3)
    )


def image_augmentation(image, datagen, number_of_augmentations):
    images = []
    image = image.reshape(1, 28, 28, 1)
    i = 0

    for x_batch in datagen.flow(image, batch_size=1):
        images.append(x_batch)
        i += 1
        if i >= number_of_augmentations:
            return images


def preprocess_data(train_images, train_labels, test_images, test_labels):
    datagen = data_generator()

    val_images = train_images[TRAIN_SET_SIZE:]
    val_labels = train_labels[TRAIN_SET_SIZE:]

    preprocessed = []

    i = 0
    for image, label in zip(train_images[:TRAIN_SET_SIZE], train_labels[:TRAIN_SET_SIZE]):
        augmented_images = image_augmentation(image, datagen, NUMBER_OF_AUGMENTATIONS)
        i += 1
        for aug_image in augmented_images:
            preprocessed.append((aug_image.reshape(28, 28, 1), label))
            if i == 13:
                plt.imshow(aug_image.reshape(28, 28))
                plt.show()
        if i == 13:
            plt.imshow(image.reshape(28, 28))
            plt.show()
        preprocessed.append((image.reshape(28, 28, 1), label))

    random.shuffle(preprocessed)

    preprocessed = list(zip(*preprocessed))
    preprocessed_x, preprocessed_y = list(preprocessed[0]), list(preprocessed[1])
    preprocessed_x = np.array(preprocessed_x) / 255

    test_images = (test_images / 255).reshape(len(test_images), 28, 28, 1)
    val_images = (val_images / 255).reshape(len(val_images), 28, 28, 1)

    return preprocessed_x, keras.utils.to_categorical(preprocessed_y), \
            val_images, keras.utils.to_categorical(val_labels), \
            test_images, keras.utils.to_categorical(test_labels),
