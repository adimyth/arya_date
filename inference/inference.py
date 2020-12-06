from glob import glob
from itertools import chain

import cv2
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.backend import ctc_decode
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from architecture import build_model
from extract_rect import ExtractRectangle


class Prediction:
    def __init__(self):
        super().__init__()
        characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        char_to_labels = StringLookup(
            vocabulary=list(characters), num_oov_indices=0, mask_token=None
        )
        self.labels_to_char = StringLookup(
            vocabulary=char_to_labels.get_vocabulary(), mask_token=None, invert=True
        )
        self.img_height = 50
        self.img_width = 250
        self.batch_size = 32
        self.max_length = 8
        self.extract = ExtractRectangle()
        self.prediction_model = build_model()
        self.prediction_model.load_weights("models/digits_ocr.h5")

    def encode_single_sample(self, img_path):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [self.img_height, self.img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Return a dict as our model is expecting two inputs
        return {"image": img}

    def get_dataset(self, image_list):
        test_dataset = tf.data.Dataset.from_tensor_slices(image_list)
        test_dataset = (
            test_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return test_dataset

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.labels_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def get_all_preds(self, dataset):
        decoded_text = []
        for batch in tqdm(dataset):
            batch_images = batch["image"]
            preds = self.prediction_model.predict(batch_images)
            pred_texts = self.decode_batch_predictions(preds)
            decoded_text.append(pred_texts)

        # flatten 2D list
        decoded_text = list(chain.from_iterable(decoded_text))
        return decoded_text

    def remove_last_one(self, label):
        idx = label.rfind("1")
        return label[:idx] + label[idx + 1 :]

    def infer(self, test_directory, output_path):
        """
        Makes prediction on test data & saves them in a csv file with image name & corresponding label predicted.

        Parameters
        ----------
        test_directory: string
            Path to directory where test images are stored

        output_path: string
            Path to store output results
        """
        # Clean & Crop Images
        print(f"[INFO] Preprocessing Images")
        test_files = sorted(list(glob(test_directory + "/*")))
        for path in tqdm(test_files):
            self.extract.process_image(path, test_directory)

        # Get Tensorflow Dataset
        print(f"[INFO] Creating DataSet")
        dataset = self.get_dataset(test_files)

        # Predict & Decode
        print(f"[INFO] Predicting & Decoding")
        decoded_text = self.get_all_preds(dataset)
        decoded_text = [
            self.remove_last_one(x) if len(x) == 9 else x for x in decoded_text
        ]

        df = pd.DataFrame({"Image": test_files, "Prediction": decoded_text})
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(Prediction)
