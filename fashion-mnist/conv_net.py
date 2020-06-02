import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

NUMBER_OF_CLASSES = 10
fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def run_training():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    x_train_full = (train_images / 255.0).astype('float32').reshape(60000, 28, 28, 1)
    x_test = (test_images / 255.0).astype('float32').reshape(10000, 28, 28, 1)

    y_train_full = keras.utils.to_categorical(train_labels)
    y_test = keras.utils.to_categorical(test_labels)

    x_train = x_train_full[:50000, ]
    x_val = x_train_full[50000:60000, ]
    y_train = y_train_full[:50000]
    y_val = y_train_full[50000:60000]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start_time = time.time()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
    training_time = int(time.time() - start_time)

    print('Training time [s]:', training_time)

    # model.evaluate(x_test, y_test, verbose=2)

    # model.save("fashion_mnist_conv_net")

    predictions = model.predict(x_test)

    num_rows = 5
    num_cols = 3
    offset = 68
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i + offset, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i + offset, predictions, test_labels)
    plt.tight_layout()
    plt.show()
