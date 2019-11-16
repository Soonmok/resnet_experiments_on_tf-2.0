from tensorflow.keras import layers
from tensorflow import keras

from utils import res_net_block, non_res_block


def get_model(num_blocks, resnet=False):
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    for i in range(num_blocks):
        if resnet:
            x = res_net_block(x, 64, 3)
        else:
            x = non_res_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    res_net_model = keras.Model(inputs, outputs)
    return res_net_model
