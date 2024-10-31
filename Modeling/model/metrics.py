# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 11:49
# @Author  : Karry Ren

""" The metrics of y_ture and y_pred.
        - r2_score: the R2 score.
        - corr_score: the Pearson correlation coefficients.
"""

import numpy as np
from sklearn import metrics


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-5):
    """ :math:`R^2` (coefficient of determination) regression score function.
    :math:`R^2 = 1 - SSR/SST`.

    Best possible score is 1.0, and it can be NEGATIVE (because the model can be arbitrarily worse,
    it need not actually be the square of a quantity R).

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param epsilon: the epsilon to avoid 0 denominator

     return:
        - r2, a number shape=()

    """

    # ---- Test the shape ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"

    # ---- Compute the SSR & SSE ---- #
    # - compute: SSR = sum((y - y_hat)^2), a number, shape=()
    ssr = np.sum((y_true - y_pred) ** 2, axis=0, dtype=np.float32)
    # - compute SST = sum((y - y_bar)^2)
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    sst = np.sum((y_true - y_bar) ** 2, axis=0, dtype=np.float32)

    # ---- Compute and Return r2 = 1 - SSR/SST ---- #
    r2 = 1 - (ssr / (sst + epsilon))
    return r2


def corr_score(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-5):
    """ The Pearson correlation coefficients.

    :math:`CORR = E[(y_true - y_true_bar)(y_pred - y_pred_bar)] / (std(y_true)*std(y_pred))`
    here we multiply `n - 1` in BOTH numerator and denominator to get:
        corr = sum((y_true - y_true_bar)(y_pred - y_pred_bar)) /
                [sqrt(sum((y_true - y_true_bar) ** 2)) * sqrt(sum((y_pred - y_pred_bar) ** 2))]

    The corr could be [-1.0, 1.0]:
        - the `0` means NO corr
        - `1` means STRONG POSITIVE corr
        - `-1` means STRONG NEGATIVE corr.

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param epsilon: the epsilon to avoid 0 denominator

    return:
        - corr, a number shape=()

    """

    # ---- Test the shape ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"

    # ---- Step 2. Compute numerator & denominator of CORR ---- #
    # compute the mean of y_ture and y_pred, shape=(num_of_samples)
    y_true_bar = np.mean(y_true, axis=0, keepdims=True)
    y_pred_bar = np.mean(y_pred, axis=0, keepdims=True)
    # compute numerator, shape=(), a number
    numerator = np.sum((y_true - y_true_bar) * (y_pred - y_pred_bar), axis=0, dtype=np.float32)
    # compute denominator, shape=(), a number
    sum_y_true_std = np.sqrt(np.sum((y_true - y_true_bar) ** 2, axis=0, dtype=np.float32))
    sum_y_pred_std = np.sqrt(np.sum((y_pred - y_pred_bar) ** 2, axis=0, dtype=np.float32))
    denominator = sum_y_true_std * sum_y_pred_std

    # ---- Compute and Return CORR score ---- #
    corr = numerator / (denominator + epsilon)
    return corr


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    """ The accuracy score of y_true and y_pred.

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)

    Attention: both y_true and y_pred are continuous.

    return:
        - acc, a number shape=()

    """

    # ---- Discretize the results ---- #
    y_true_dis = np.sign(y_true)
    y_pred_dis = np.sign(y_pred)

    # ---- Compute the acc ---- #
    return metrics.accuracy_score(y_true=y_true_dis, y_pred=y_pred_dis)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """ The F1 score of y_true and y_pred.

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)

    Attention: both y_true and y_pred are continuous.

    return:
        - f1, a number shape=()

    """

    # ---- Discretize the results ---- #
    y_true_dis = np.sign(y_true)
    y_pred_dis = np.sign(y_pred)

    # ---- Compute the F1 ---- #
    return metrics.f1_score(y_true=y_true_dis, y_pred=y_pred_dis, average="macro")


if __name__ == "__main__":
    y_true = np.array([-1, 0, 0, 1])
    y_pred = np.array([-11, 0, 3, 1])

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    print("r2 = ", r2)
    corr = corr_score(y_true=y_true, y_pred=y_pred)
    print("corr = ", corr)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print("acc = ", acc)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print("f1 = ", f1)
