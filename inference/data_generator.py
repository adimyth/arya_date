import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates batches from a given dataset.

    Args:
        data: training or validation data
        char_map: dictionary mapping char to labels
        batch_size: size of a single batch
        img_width: width of the resized
        img_height: height of the resized
        downsample_factor: by what factor did the CNN downsample the images
        max_length: maximum length of any captcha
        shuffle: whether to shuffle data or not after each epoch
    Returns:
        batch_inputs: a dictionary containing batch inputs
    """

    def __init__(self, data, img_height, img_width, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.indices = np.arange(len(data))
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # 1. Get the next batch indices
        curr_batch_idx = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        # 2. This isn't necessary but it can help us save some memory
        # as not all batches the last batch may not have elements
        # equal to the batch_size
        batch_len = len(curr_batch_idx)

        # 3. Instantiate batch arrays
        batch_images = np.ones(
            (batch_len, self.img_width, self.img_height, 1), dtype=np.float32
        )

        for j, idx in enumerate(curr_batch_idx):
            # 1. Get the image and transpose it
            img = self.data[idx].T
            # 2. Add extra dimenison
            img = np.expand_dims(img, axis=-1)
            batch_images[j] = img

        return batch_images
