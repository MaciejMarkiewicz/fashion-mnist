from tensorflow import keras

NUMBER_OF_CLASSES = 10
BATCH_SIZE = 512
EPOCHS = 150


def run_training(train_images, train_labels, val_images, val_labels):

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='Same', input_shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='Same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(train_images, train_labels,
              validation_data=(val_images, val_labels),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS
              )

    model.save('trained_model/model.h5')
