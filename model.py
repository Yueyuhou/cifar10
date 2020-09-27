import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Model_1():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), padding='same', data_format='channels_first',
                          activation='relu')(inputs)
        x = layers.Conv2D(32, (3, 3), data_format='channels_first', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3, 3), padding='same', data_format='channels_first',
                          activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), data_format='channels_first', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='model_1')
        return model
