# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 10:58
# @Author  : Karry Ren

""" The dataset of factors.

You should make your data directory like:
root_path/
    ├── lag_{time_steps}_factor_data.npz

"""

from torch.utils import data
import numpy as np


class FactoDataset(data.Dataset):
    """ The torch.Dataset of factor dataset. """

    def __init__(self, root_path: str, time_steps: int = 1):
        """ The init function of Factor Dataset.

        :param root_path: the root path of UCI electricity dataset
        :param time_steps: the time steps (lag steps)

        """

        # ---- Read the data ---- #
        factor_data_array = np.load(f"{root_path}/lag_{time_steps}_factor_data.npz")

        # ---- Get the feature and label (make sure of the dtype) ---- #
        self.feature_array = factor_data_array["feature"].astype(np.float32)
        self.label_array = factor_data_array["label"].astype(np.float32)
        assert self.feature_array.shape[0] == self.label_array.shape[0], "Data ERROR !"

    def __len__(self):
        """ Get the length of dataset. """

        return self.feature_array.shape[0]

    def __getitem__(self, idx: int):
        """ Get the item based on idx, and lag the item.

        return: item_data (one lagged day sample of one client)
            - `feature`: the feature, shape=(time_steps, feature_dim)
            - `label`: the return label, shape=(1, )

        """

        # ---- Construct item data and return ---- #
        item_data = {"feature": self.feature_array[idx], "label": self.label_array[idx]}
        return item_data


if __name__ == "__main__":  # a demo using FactorDataset
    FACTOR_DATASET_PATH = "../../../Data"
    data_set = FactoDataset(FACTOR_DATASET_PATH, time_steps=1)
    for i in range(0, len(data_set) - 1):
        item_data = data_set[i]
        print(item_data["feature"])
        print(item_data["label"])
        break
