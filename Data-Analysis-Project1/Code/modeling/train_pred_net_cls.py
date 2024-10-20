# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 12:06
# @Author  : Karry Ren


""" Training and Prediction code. (classification) """

import os
import logging
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from utils import fix_random_seed
import config as config
from factor_dataset import FactoDataset
from model.mlp import MLP_Net
from model.loss import CE_Loss
from sklearn import metrics


def train_valid_model() -> None:
    """ Train & Valid Model. """

    logging.info(f"***************** RUN TRAIN&VALID MODEL *****************")

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER ! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    # make the dataset and dataloader of training
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    train_dataset = FactoDataset(root_path=config.FACTOR_DATA_PATH, time_steps=config.TIME_STEPS)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)  # the train dataloader
    logging.info(f"**** VALID DATASET & DATALOADER ****")
    valid_dataset = FactoDataset(root_path=config.FACTOR_DATA_PATH, time_steps=config.TIME_STEPS)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    model = MLP_Net(input_size=config.FACTOR_NUM, device=device)
    # the loss function
    criterion = CE_Loss(reduction=config.LOSS_REDUCTION)
    # the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # ---- Start Train and Valid ---- #
    # init the metric dict of all epochs
    epoch_metric = {
        "train_loss": np.zeros(config.EPOCHS), "valid_loss": np.zeros(config.EPOCHS),
        "valid_ACC": np.zeros(config.EPOCHS), "valid_F1": np.zeros(config.EPOCHS)
    }
    # train model epoch by epoch
    logging.info(f"***************** BEGIN TRAINING ! *****************")
    # start train and valid during train
    for epoch in tqdm(range(config.EPOCHS)):
        # start timer for one epoch
        t_start = datetime.now()
        # set the array for one epoch to store (all empty)
        train_loss_one_epoch, valid_loss_one_epoch = [], []
        train_dataset_len, valid_dataset_len = len(train_dataset), len(valid_dataset)
        train_preds_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_labels_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        valid_preds_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_labels_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        # - train model
        last_step = 0
        model.train()
        for batch_data in tqdm(train_loader):
            # move data to device
            features, labels = batch_data["feature"].to(device=device), batch_data["sign_label"].to(device=device)
            # zero_grad, forward, compute loss, backward and optimize
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(y_true=labels, y_pred=preds)
            loss.backward()
            optimizer.step()
            # note the loss of training in one iter
            train_loss_one_epoch.append(loss.item())
            # note the result in one iter
            now_step = last_step + preds.shape[0]
            train_preds_one_epoch[last_step:now_step] = torch.argmax(torch.softmax(preds, dim=1), dim=1).detach()
            train_labels_one_epoch[last_step:now_step] = labels.detach()
            last_step = now_step
        # note the loss and metrics for one epoch of TRAINING
        epoch_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
        # - valid model
        last_step = 0
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                # move data to device
                features, labels = batch_data["feature"].to(device=device), batch_data["sign_label"].to(device=device)
                # forward to compute outputs, different model have different loss
                preds = model(features)
                loss = criterion(y_true=labels, y_pred=preds)
                # note the loss of valid in one iter
                valid_loss_one_epoch.append(loss.item())
                # doc the result in one iter
                now_step = last_step + preds.shape[0]
                valid_preds_one_epoch[last_step:now_step] = torch.argmax(torch.softmax(preds, dim=1), dim=1).detach()
                valid_labels_one_epoch[last_step:now_step] = labels.detach()
                last_step = now_step
        # note the loss and all metrics for one epoch of VALID
        epoch_metric["valid_loss"][epoch] = np.mean(valid_loss_one_epoch)
        epoch_metric["valid_ACC"][epoch] = metrics.accuracy_score(
            y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy()
        )
        epoch_metric["valid_F1"][epoch] = metrics.f1_score(
            y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy(), average="micro"
        )

        # save model&model_config and metrics
        torch.save(model, f"{config.MODEL_SAVE_PATH}/model_pytorch_epoch_{epoch}")
        pd.DataFrame(epoch_metric).to_csv(f"{config.MODEL_SAVE_PATH}/model_metric.csv")

        # write metric log
        dt = datetime.now() - t_start
        logging.info(f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, "
                     f"{['%s:%.4f ' % (key, value[epoch]) for key, value in epoch_metric.items()]}")
    # draw figure of train and valid metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epoch_metric["train_loss"], label="train loss", color="g")
    plt.plot(epoch_metric["valid_loss"], label="valid loss", color="b")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(epoch_metric["valid_ACC"], label="valid ACC", color="b")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(epoch_metric["valid_F1"], label="Valid F1", color="b")
    plt.legend()
    plt.savefig(f"{config.SAVE_PATH}training_steps.png", dpi=200, bbox_inches="tight")
    logging.info("***************** TRAINING OVER ! *****************")


if __name__ == "__main__":
    # ---- Prepare some environments for training and prediction ---- #
    # fix the random seed
    fix_random_seed(seed=config.RANDOM_SEED)
    # build up the save directory
    if not os.path.exists(f"{config.SAVE_PATH}"):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    # construct the train&valid log file
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # ---- Step 1. Train & Valid model ---- #
    train_valid_model()
