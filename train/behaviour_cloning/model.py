from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Input, Flatten, Dense
from keras.models import Model

input_image_height = 384
input_image_width = 512
input_image_channels = 3


def create_model():
    image_in = Input(shape=(input_image_height, input_image_width, input_image_channels))
    x = image_in

    x = Convolution2D(24, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(strides=(3, 3))(x)

    x = Convolution2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(strides=(3, 3))(x)

    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(strides=(3, 3))(x)

    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(strides=(3, 3))(x)

    x = Flatten()(x)

    x = Dense(50, activation='relu')(x)
    angle_out = Dense(1, activation='linear', name='angle_out')(x)

    model = Model(inputs=[image_in], outputs=[angle_out])
    model.compile(optimizer='adam', loss={'angle_out': 'mse'})
    return model
