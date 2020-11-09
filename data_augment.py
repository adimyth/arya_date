import itertools

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm


class DataAugmenter:
    def __init__(self):
        super(DataAugmenter).__init__()
        self.aug_path = []
        self.aug_labels = []
        train_data = pd.read_csv("data/raw/train/train_data.csv")[["tag", "label"]]
        train_data["label"] = train_data["label"].astype(str).str.zfill(8)
        self.train_data = dict(
            zip(train_data["tag"].tolist(), train_data["label"].tolist())
        )
        self.count = 10000

    def stitch_images_horizontally(
        self,
        inp_files,
        save_path,
        height,
        width,
        digits1=4,
        digits2=4,
        comb_type="type1",
    ):
        """
        Crops & stitches two images.
        Crops initial `digit1` & initial `digit2` digits from corresponding images & stitches them horizontally

        Arguments
        ---------
        inp_files - list
            List of image pair paths

        height - int
            Image height for both images.

        width - int
            Image width for both images

        digit1 - int
            Number of digits to take from first image

        digit2 - int
            Number of digits to take from second image

        comb_type - str
            Denotes combination type.
            type1 - Initial part of both images
            type2 - Last part of first image & initial part of second image
            type3 - Initial part of first image & last part of second image
            type4 - Last part of both images
        """

        images = [Image.open(x) for x in inp_files]

        w1 = int(width * (digits1 / 8))
        w2 = int(width * (digits2 / 8))

        if comb_type == "type1":
            img1_crop = images[0].crop((0, 0, w1, height))
            img2_crop = images[1].crop((0, 0, w2, height))
            images = [img1_crop, img2_crop]
        elif comb_type == "type2":
            img1_crop = images[0].crop((w1, 0, width, height))
            img2_crop = images[1].crop((0, 0, w2, height))
            images = [img1_crop, img2_crop]
        elif comb_type == "type3":
            img1_crop = images[0].crop((0, 0, w1, height))
            img2_crop = images[1].crop((w2, 0, width, height))
            images = [img1_crop, img2_crop]
        elif comb_type == "type4":
            img1_crop = images[0].crop((w1, 0, width, height))
            img2_crop = images[1].crop((w2, 0, width, height))
            images = [img1_crop, img2_crop]

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = min(heights)

        new_im = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(save_path)

    def process(self):
        df = pd.read_csv("cropped_image_info.csv")
        df["area"] = df["height"].multiply(df["width"])
        df = df.loc[df["path"].str.contains("train")]
        df["tag"] = df["path"].apply(lambda x: int(x.split("/")[-1].split(".")[0]))
        df = df.sort_values(by="tag")
        df["path"] = df["path"].str.replace("raw", "processed")
        temp = dict(zip(df["tag"].tolist(), df["path"].tolist()))

        # find matching areas
        matching_areas = df["area"].value_counts()
        matching_areas = matching_areas[matching_areas > 2].index.tolist()
        # loop over matching areas
        for area in tqdm(matching_areas):
            # get height & width
            height = df.loc[df["area"] == area]["height"].iloc[0]
            width = df.loc[df["area"] == area]["width"].iloc[0]
            # create permutation based on tag
            all_combinations = list(
                itertools.combinations(
                    df.loc[df["area"] == area]["tag"].tolist()[:3], 2
                )
            )
            # for each combination
            for comb in all_combinations:
                tag1, tag2 = comb[0], comb[1]
                image1 = temp[tag1]
                image2 = temp[tag2]
                label1 = self.train_data[tag1]
                label2 = self.train_data[tag2]

                # variation based on number of digits per image - type1
                for digit1 in range(1, 5):
                    digit2 = 8 - digit1
                    self.count += 1
                    path = f"data/processed/train/{self.count}.png"
                    label = str(label1)[:digit1] + str(label2)[:digit2]
                    self.stitch_images_horizontally(
                        [image1, image2], path, height, width, digit1, digit2
                    )

                    self.aug_labels.append(label)
                    self.aug_path.append(path)

                # different types
                for comb_type in ["type2", "type3", "type4"]:
                    self.count += 1
                    path = f"data/processed/train/{self.count}.png"
                    if comb_type == "type2":
                        label = str(label1)[4:] + str(label2)[:4]
                    elif comb_type == "type3":
                        label = str(label1)[:4] + str(label2)[4:]
                    elif comb_type == "type4":
                        label = str(label1)[4:] + str(label2)[4:]

                    self.stitch_images_horizontally(
                        [image1, image2], path, height, width, comb_type=comb_type
                    )
                    self.aug_labels.append(label)
                    self.aug_path.append(path)

    def add_reverse(self):
        df = pd.read_csv("cropped_image_info.csv")
        df = df.loc[df["path"].str.contains("train")]
        df["tag"] = df["path"].apply(lambda x: int(x.split("/")[-1].split(".")[0]))
        df = df.sort_values(by="tag")
        df["path"] = df["path"].str.replace("raw", "processed")
        image_temp = dict(zip(df["tag"].tolist(), df["path"].tolist()))
        height_temp = dict(zip(df["tag"].tolist(), df["height"].tolist()))
        width_temp = dict(zip(df["tag"].tolist(), df["width"].tolist()))
        actual_df = pd.read_csv("data/raw/train/train_data.csv")
        actual_df = actual_df[["tag", "label"]]

        for tag in tqdm(df["tag"].tolist()):
            image = image_temp[tag]
            height = height_temp[tag]
            width = width_temp[tag]
            act_label = str(
                actual_df.loc[actual_df["tag"] == tag]["label"].tolist()[0]
            ).zfill(8)

            # variation based on number of digits per image - type1
            for digit1 in range(1, 7):
                digit2 = 8 - digit1
                self.count += 1
                path = f"data/processed/train/{self.count}.png"
                label = str(act_label)[:digit1] + str(act_label)[:digit2]
                self.stitch_images_horizontally(
                    [image, image], path, height, width, digit1, digit2
                )

                self.aug_labels.append(label)
                self.aug_path.append(path)
        updated_train_data = pd.DataFrame.from_dict(
            {"tag": self.aug_path, "label": self.aug_labels}
        )
        updated_train_data.to_csv("updated_train_data.csv", index=False)


if __name__ == "__main__":
    augment = DataAugmenter()
    augment.process()
    x = augment.count
    print(f"# AUGMENTED IMAGES: {x - 10000}")
    augment.add_reverse()
    print(f"# REVERSE IMAGES: {augment.count - x}")
