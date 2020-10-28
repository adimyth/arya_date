from glob import glob
import random

import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class ImageSplitter:
    def __init__(self):
        super().__init__()

    def remove_whitespace(self, img):
        # https://stackoverflow.com/questions/48395434/how-to-crop-or-remove-white-background-from-an-image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

        cnts = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        x, y, w, h = cv.boundingRect(cnt)
        dst = img[y : y + h, x : x + w]
        return dst

    def split_image(self, path, label):
        label = str(label).zfill(8)
        img = cv.imread(cv.samples.findFile(path))
        img = self.remove_whitespace(img)

        height, width = img.shape[:2]
        start, end = 0, 0
        cw = int(width / 8)
        for idx in range(8):
            crop = img[0:height, idx * cw : (idx + 1) * cw]
            cv.imwrite(
                f"data/digits/train/{int(label[idx])}/{random.randint(0, 1000)}.png",
                crop,
            )


if __name__ == "__main__":
    base = "data/interim/train/"
    train_files = glob(base + "/*.png")
    df = pd.read_csv(base + "corrected_train_data.csv")
    train_labels = [
        df[df["tag"] == int(x.split("/")[-1].split(".")[0])]["label"].values[0]
        for x in train_files
    ]

    extract = ImageSplitter()
    for path, label in tqdm(dict(zip(train_files, train_labels)).items()):
        extract.split_image(path, label)
