import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

NUMBER_OF_CLASSES = 10
BATCH_SIZE = 512
EPOCHS = 150
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def run_training(train_images, train_labels, val_images, val_labels, test_images, test_labels):

    x_train = train_images
    x_test = test_images

    y_train = train_labels
    y_test = test_labels

    x_val = val_images
    y_val = val_labels

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='Same', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='Same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS
              )

    # model.evaluate(x_test, y_test, verbose=2)

    model.save("model.h5")


