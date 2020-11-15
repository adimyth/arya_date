import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv2D,
    Dense,
    Input,
    Layer,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.models import Model


class CTCLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # On test time, just return the computed loss
        return loss


def build_model():
    img_width, img_height = (250, 50)
    characters = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Inputs to the model
    input_img = Input(
        shape=(img_width, img_height, 1), name="input_data", dtype="float32"
    )

    # First conv block
    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing it to RNNs
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(64, activation="relu", name="dense1")(x)

    # RNNs
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Predictions
    output = Dense(
        len(characters) + 1,
        activation="softmax",
        name="dense2",
        kernel_initializer="he_normal",
    )(x)

    # Define the model
    model = Model(
        inputs=input_img,
        outputs=output,
        name="ocr_model_v1",
    )

    return model
