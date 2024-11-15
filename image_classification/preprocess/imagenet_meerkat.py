import os
import subprocess
from typing import Dict

import numpy as np
import pandas as pd

# import meerkat as mk

class mk:

    class DataFrame:

        @classmethod
        def from_pandas(
            cls,
            df: pd.DataFrame,
            index: bool = True,
            # primary_key = None,
        ) -> pd.DataFrame:
            """Create a Meerkat DataFrame from a Pandas DataFrame.

            .. warning::

                In Meerkat, column names must be strings, so non-string column
                names in the Pandas DataFrame will be converted.

            Args:
                df: The Pandas DataFrame to convert.
                index: Whether to include the index of the Pandas DataFrame as a
                    column in the Meerkat DataFrame.
                primary_key: The name of the column to use as the primary key.
                    If `index` is True and `primary_key` is None, the index will
                    be used as the primary key. If `index` is False, then no
                    primary key will be set. Optional default is None.

            Returns:
                DataFrame: The Meerkat DataFrame.
            """

            # column names must be str in meerkat
            df = df.rename(mapper=str, axis="columns")
            pandas_df = df

            # df = cls.from_batch(
            #     pandas_df.to_dict("series"),
            # )
            df = dict(pandas_df.to_dict("series"),)

            if index:
                df["index"] = pandas_df.index.to_series()
            #     if primary_key is None:
            #         df.set_primary_key("index", inplace=True)

            # if primary_key is not None:
            #     df.set_primary_key(primary_key, inplace=True)

            return df
        
    class ImageColumn:
        @classmethod
        def from_filepaths(cls, paths):
            return paths

# from ..abstract import DatasetBuilder
# from ..info import DatasetInfo
# from ..registry import datasets


# @datasets.register()
class imagenet: #(DatasetBuilder):
    VERSIONS = ["ilsvrc2012"]

    info = dict( #DatasetInfo(
        name="imagenet",
        full_name="ImageNet",
        # flake8: noqa
        description="ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images..",
        homepage="https://www.image-net.org/",
        tags=["image", "classification"],
        citation=(
            "@inproceedings{imagenet_cvpr09,"
            "AUTHOR = {Deng, J. and Dong, W. and Socher, R. and Li, L.-J. and Li, K. and Fei-Fei, L.},"
            "TITLE = {{ImageNet: A Large-Scale Hierarchical Image Database}},"
            "BOOKTITLE = {CVPR09},"
            "YEAR = {2009},"
            'BIBSOURCE = "http://www.image-net.org/papers/imagenet_cvpr09.bib"}'
        ),
    )

    def build(self):
        paths = pd.read_csv(
            os.path.join(self.dataset_dir, "ILSVRC/ImageSets/CLS-LOC/train_cls.txt"),
            delimiter=" ",
            names=["path", "idx"],
        )["path"]
        train_df = paths.str.extract(r"(?P<synset>.*)/(?P<image_id>.*)")

        train_df["path"] = paths.apply(
            lambda x: os.path.join(
                self.dataset_dir, "ILSVRC/Data/CLS-LOC/train", f"{x}.JPEG"
            )
        )
        train_df["split"] = "train"

        # load validation data
        valid_df = pd.read_csv(
            os.path.join(self.dataset_dir, "LOC_val_solution.csv")
        ).rename(columns={"ImageId": "image_id"})
        valid_df["synset"] = valid_df["PredictionString"].str.split(" ", expand=True)[0]
        valid_df["path"] = valid_df["image_id"].apply(
            lambda x: os.path.join(
                self.dataset_dir, "ILSVRC/Data/CLS-LOC/val", f"{x}.JPEG"
            )
        )
        valid_df["split"] = "valid"

        df = mk.DataFrame.from_pandas(
            pd.concat([train_df, valid_df.drop(columns="PredictionString")])
        )
        df["image"] = mk.ImageColumn.from_filepaths(df["path"])

        # mapping from synset to english
        with open(os.path.join(self.dataset_dir, "LOC_synset_mapping.txt")) as f:
            lines = f.read().splitlines()
        df = (
            pd.Series(lines)
            .str.split(" ", expand=True, n=1)
            .rename(columns={0: "synset", 1: "name"})
        )
        mapping_df = mk.DataFrame.from_pandas(df)

        # torchvision models use class indices corresponding to the order of the
        # LOC_synset_mapping.txt file, which we confirmed using the mapping provided here
        # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        mapping_df["class_idx"] = np.arange(len(mapping_df))
        df = df.merge(mapping_df, how="left", on="synset")

        return df

    def download(self):
        curr_dir = os.getcwd()
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.chdir(self.dataset_dir)
        subprocess.run(
            args=[
                "kaggle competitions download "
                "-c imagenet-object-localization-challenge",
            ],
            shell=True,
            check=True,
        )
        subprocess.run(["unzip", "imagenet-object-localization-challenge.zip"])
        subprocess.run(
            ["tar", "-xzvf", "imagenet_object_localization_patched2019.tar.gz"]
        )
        os.chdir(curr_dir)


def build_imagenet_dfs(
    dataset_dir: str, download: bool = False
) -> Dict[str, mk.DataFrame]:
    if download:
        curr_dir = os.getcwd()
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)
        # subprocess.run(
        #     args=[
        #         "kaggle competitions download "
        #         "-c imagenet-object-localization-challenge",
        #     ],
        #     shell=True,
        #     check=True,
        # )
        # subprocess.run(["unzip", "imagenet-object-localization-challenge.zip"])
        subprocess.run(
            ["tar", "-xzvf", "imagenet_object_localization_patched2019.tar.gz"]
        )
        os.chdir(curr_dir)

    # load training data
    paths = pd.read_csv(
        os.path.join(dataset_dir, "ILSVRC/ImageSets/CLS-LOC/train_cls.txt"),
        delimiter=" ",
        names=["path", "idx"],
    )["path"]
    train_df = paths.str.extract(r"(?P<synset>.*)/(?P<image_id>.*)")

    train_df["path"] = paths.apply(
        lambda x: os.path.join(dataset_dir, "ILSVRC/Data/CLS-LOC/train", f"{x}.JPEG")
    )
    train_df["split"] = "train"

    # load validation data
    valid_df = pd.read_csv(os.path.join(dataset_dir, "LOC_val_solution.csv")).rename(
        columns={"ImageId": "image_id"}
    )
    valid_df["synset"] = valid_df["PredictionString"].str.split(" ", expand=True)[0]
    valid_df["path"] = valid_df["image_id"].apply(
        lambda x: os.path.join(dataset_dir, "ILSVRC/Data/CLS-LOC/val", f"{x}.JPEG")
    )
    valid_df["split"] = "valid"

    # df = mk.DataFrame.from_pandas(
    #     pd.concat([train_df, valid_df.drop(columns="PredictionString")])
    # )
    df = pd.concat([train_df, valid_df.drop(columns="PredictionString")])

    df["image"] = mk.ImageColumn.from_filepaths(df["path"])

    # mapping from synset to english
    with open(os.path.join(dataset_dir, "LOC_synset_mapping.txt")) as f:
        lines = f.read().splitlines()
    mapping_df = (
        pd.Series(lines)
        .str.split(" ", expand=True, n=1)
        .rename(columns={0: "synset", 1: "name"})
    )
    # mapping_df = mk.DataFrame.from_pandas(mapping_df)

    # torchvision models use class indices corresponding to the order of the
    # LOC_synset_mapping.txt file, which we confirmed using the mapping provided here
    # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    mapping_df["class_idx"] = np.arange(len(mapping_df))
    df = df.merge(mapping_df, how="left", on="synset")

    return df


if __name__ == "__main__":
    build_imagenet_dfs("/capstor/scratch/cscs/dealmeih/ds/mlperf/data/image_classification/", download=False)
    print('All done')