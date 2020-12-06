import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Input,
    Layer,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.models import Model


def build_model():
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    img_width, img_height = (250, 50)

    # Input to the model
    input_img = Input(
        shape=(img_width, img_height, 1), dtype="float32"
    )

    # First conv block
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Second conv block
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Third conv block
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization(name="bn_3")(x)

    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller.
    new_shape = ((img_width // 4), (img_height // 4) * 256)
    x = Reshape(target_shape=new_shape)(x)
    x = Dense(128, activation="relu")(x)

    # RNNs
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Predictions
    output = Dense(
        len(characters) + 1,
        activation="softmax"
    )(x)

    # Define the model
    model = Model(
        inputs=input_img,
        outputs=output
    )

    return model
