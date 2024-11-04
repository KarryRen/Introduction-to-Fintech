# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 10:58
# @Author  : Karry Ren

""" The dataset of factors.

You should make your data directory like:
root_path/
    ├── lag_1
        ├── stock_1.npz
        ├── stock_2.npz
        ├── ...
        └── stock_n.npz
    ├── lag_2
    ├── ...
    ├── lag_n
    ├── split
        ├── lag_1
            ├── stock_1.npz
            ├── stock_2.npz
            ├── ...
            └── stock_n.npz
        ├── lag_2
        ├── ...
        └── lag_n

"""
import pandas as pd
from torch.utils import data
import numpy as np
from typing import List
import os


class FactoDataset(data.Dataset):
    """ The torch.Dataset of factor dataset. """

    def __init__(self, root_path: str, time_steps: int = 1, stock_file_list: List[str] = None, data_type: str = None):
        """ The init function of Factor Dataset.

        :param root_path: the root path of dataset
        :param time_steps: the time steps (lag steps)
        :param stock_file_list: the stock list, if None, use all stocks (format should be `stock_code.npz`)
        :param data_type: the data type, if None, use all data as train & valid & test else:
            - `train`
            - `valid`
            - `test`

        """

        # ---- Read the data ---- #
        if stock_file_list is None:
            stock_file_list = sorted(os.listdir(f"{root_path}/lag_{time_steps}"))

        # ---- Build up the type data ---- #
        if data_type is None:
            data_path = f"{root_path}/lag_{time_steps}"
        else:
            data_path = f"{root_path}/split/lag_{time_steps}/{data_type}"

        # ---- Get the feature and label (make sure of the dtype) ---- #
        # read
        feature_array_list, label_array_list = [], []
        for stock_file in stock_file_list:
            feature_array_list.append(np.load(f"{data_path}/{stock_file}")["feature"])
            label_array_list.append(np.load(f"{data_path}/{stock_file}")["label"])
        # concat
        self.feature_array = np.concatenate(feature_array_list, axis=0)
        self.label_array = np.concatenate(label_array_list, axis=0)
        print(self.label_array.shape)
        assert self.feature_array.shape[0] == self.label_array.shape[0], "Data ERROR !"

        # ---- Build up the sign label ---- #
        self.sign_label_array = np.sign(self.label_array).astype(np.int64)
        self.sign_label_array[self.sign_label_array == 1] = 2
        self.sign_label_array[self.sign_label_array == 0] = 1
        self.sign_label_array[self.sign_label_array == -1] = 0

    def __len__(self):
        """ Get the length of dataset. """

        return self.feature_array.shape[0]

    def __getitem__(self, idx: int):
        """ Get the item based on idx, and lag the item.

        return: item_data (one lagged day sample of one client)
            - `feature`: the feature, shape=(time_steps, feature_dim)
            - `label`: the return label, shape=(1, )
            - `sign_label`: the sign of return label, a number

        """

        # ---- Construct item data and return ---- #
        item_data = {"feature": self.feature_array[idx], "label": self.label_array[idx], "sign_label": self.sign_label_array[idx, 0]}
        return item_data


class TextFactoDataset(data.Dataset):
    """ The torch.Dataset of text factor dataset. """

    def __init__(self, file_path: str, label_type: str, feature_type: str, data_type: str = "train"):
        """ The init function of Factor Dataset.

        :param file_path: the root path of dataset
        :param label_type: the type of our label, you can choose
            - `Label_1`: the label of 1 day
            - `Label_3`: the label of 3 days
            - `Label_10`: the label of 10 days`
        :param feature_type: the type of our feature, you can choose
            - `embedding`: raw feature
            - `PCA` : the feature after PCA
            - `Lasso`: the feature after Lasso
            - `XGB: the feature` after XGB
        :param data_type: the type of our data, you can choose
            - `train`: train data
            - `valid`: valid data
            - `test`: test data

        """

        # ---- Read the data ---- #
        text_factor_data_df = pd.read_pickle(file_path)
        text_factor_data_df_train = text_factor_data_df[text_factor_data_df["matched_date"] < "20230601"]
        text_factor_data_df_vt = text_factor_data_df[text_factor_data_df["matched_date"] >= "20230601"]
        text_factor_data_df_valid = text_factor_data_df_vt[text_factor_data_df_vt["matched_date"] < "20240101"]
        text_factor_data_df_test = text_factor_data_df_vt[text_factor_data_df_vt["matched_date"] >= "20240101"]

        # ---- Set to feature and label ---- #
        # split
        if data_type == "train":
            text_factor_data_df = text_factor_data_df_train
        elif data_type == "valid":
            text_factor_data_df = text_factor_data_df_valid
        elif data_type == "test":
            text_factor_data_df = text_factor_data_df_test
        else:
            raise TypeError
        # feature
        text_feature_list = []
        for text_feature in text_factor_data_df[feature_type]:
            text_feature_list.append([text_feature])
        self.feature_array = np.concatenate(text_feature_list, axis=0, dtype=np.float32)
        self.feature_array = np.expand_dims(self.feature_array, axis=1)
        # label
        self.label_array = np.array(text_factor_data_df[label_type], dtype=np.float32)
        self.label_array = np.expand_dims(self.label_array, axis=1)

        # ---- Build up the sign label ---- #
        self.sign_label_array = np.sign(self.label_array).astype(np.int64)
        self.sign_label_array[self.sign_label_array == 1] = 2
        self.sign_label_array[self.sign_label_array == 0] = 1
        self.sign_label_array[self.sign_label_array == -1] = 0

    def __len__(self):
        """ Get the length of dataset. """

        return self.feature_array.shape[0]

    def __getitem__(self, idx: int):
        """ Get the item based on idx, and lag the item.

        return: item_data (one lagged day sample of one client)
            - `feature`: the feature, shape=(time_steps, feature_dim)
            - `label`: the return label, shape=(1, )
            - `sign_label`: the sign of return label, a number

        """

        # ---- Construct item data and return ---- #
        item_data = {"feature": self.feature_array[idx], "label": self.label_array[idx], "sign_label": self.sign_label_array[idx, 0]}
        return item_data


if __name__ == "__main__":  # a demo using FactorDataset
    # FACTOR_DATASET_PATH = "../Data/processed_factors"
    # data_set = FactoDataset(FACTOR_DATASET_PATH, time_steps=1, stock_file_list=["000009.sz.npz"], data_type="test")
    # print(len(data_set))
    # for i in range(0, len(data_set) - 1):
    #     item_data = data_set[i]
    #     print(item_data["feature"])
    #     print(item_data["label"])
    #     print(item_data["sign_label"])
    #     break

    TEXT_DATA_FILE = "../Data/text_factors/processed_text_factors.pkl"
    train_data_set = TextFactoDataset(TEXT_DATA_FILE, label_type="Label_1", feature_type="embedding", data_type="train")
    print(len(train_data_set))
    valid_data_set = TextFactoDataset(TEXT_DATA_FILE, label_type="Label_1", feature_type="embedding", data_type="valid")
    print(len(valid_data_set))
    test_data_set = TextFactoDataset(TEXT_DATA_FILE, label_type="Label_1", feature_type="embedding", data_type="test")
    print(len(test_data_set))
