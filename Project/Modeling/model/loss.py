# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 11:46
# @Author  : Karry Ren

""" The loss functions. """

import torch
import torch.nn.functional as F


class MSE_Loss:
    """ Compute the MSE loss.

    loss = reduction((y_true - y_pred)^2)

    """

    def __init__(self, reduction: str = "mean"):
        """ Init function of the MSE Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction

        """

        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """ Call function of the MSE Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Compute the loss ---- #
        if self.reduction == "mean":
            # compute mse loss (`mean`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.mean(mse_sample_loss)  # mean the loss
            batch_loss = mse_loss
        elif self.reduction == "sum":
            # compute mse loss (`sum`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum(mse_sample_loss)  # sum the loss
            batch_loss = mse_loss
        else:
            raise TypeError(self.reduction)

        # ---- Return loss ---- #
        return batch_loss


class CE_Loss:
    """ Compute the Cross Entropy loss.

    loss = reduction((y_true - y_pred)^2)

    """

    def __init__(self, reduction: str = "mean"):
        """ Init function of the MSE Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction

        """

        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """ Call function of the MSE Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Compute the loss ---- #
        batch_loss = F.cross_entropy(target=y_true, input=y_pred, reduction=self.reduction)

        # ---- Return loss ---- #
        return batch_loss


if __name__ == "__main__":
    y_true = torch.tensor([0, 2])
    y_pred = torch.tensor([[10, 1000, -1000], [-1000, -1000, 10]], dtype=torch.float32)
    print(y_true.shape, y_pred.shape)
    ce_loss = CE_Loss()
    print(ce_loss(y_true, y_pred))
