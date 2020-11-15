from glob import glob
from itertools import chain

import cv2
import fire
import numpy as np
import pandas as pd
from architecture import build_model
from data_generator import DataGenerator
from tensorflow.keras.backend import ctc_decode
from tqdm import tqdm

from extract_rect import ExtractRectangle


class Prediction:
    def __init__(self):
        super().__init__()
        characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        char_to_labels = {char: idx for idx, char in enumerate(characters)}
        self.labels_to_char = {val: key for key, val in char_to_labels.items()}
        self.img_height = 50
        self.img_width = 250
        self.extract = ExtractRectangle()
        self.prediction_model = build_model()
        self.prediction_model.load_weights("models/digits_ocr.h5")

    def generate_arrays(self, image_list):
        num_items = len(image_list)
        images = np.zeros(
            (num_items, self.img_height, self.img_width), dtype=np.float32
        )

        for i, path in enumerate(image_list):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = (img / 255.0).astype(np.float32)
            images[i, :, :] = img
        return images

    def decode_batch_predictions(self, pred):
        pred = pred[:, :-2]
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # greedy search
        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for res in results.numpy():
            outstr = ""
            for c in res:
                if c < 10 and c >= 0:
                    outstr += self.labels_to_char[c]
            output_text.append(outstr)
        return output_text

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

        # Numpy Image array
        print(f"[INFO] Loading Images")
        data = self.generate_arrays(test_files)

        # Get DataGenerator
        print(f"[INFO] Creating Data Generator")
        data_generator = DataGenerator(data, self.img_height, self.img_width)

        # Predict & Decode
        print(f"[INFO] Predicting & Decoding")
        decoded_text = []
        for X_data in tqdm(data_generator):
            bs = X_data.shape[0]
            preds = self.prediction_model.predict(X_data)
            pred_texts = self.decode_batch_predictions(preds)
            decoded_text.append([pred_texts[i] for i in range(bs)])
        # flatten 2D list
        decoded_text = list(chain.from_iterable(decoded_text))
        decoded_text = [
            self.remove_last_one(x) if len(x) == 9 else x for x in decoded_text
        ]

        df = pd.DataFrame({"Image": test_files, "Prediction": decoded_text})
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(Prediction)
