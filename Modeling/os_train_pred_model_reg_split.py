# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 16:50
# @Author  : Karry Ren

""" Training and Prediction code. (regression) for overall stock. """

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
from model.nets import MLP_Net, Big_MLP_Net, GRU_Net, LSTM_Net, Conv_Net, Transformer_Net
from model.loss import MSE_Loss
from model.metrics import r2_score, corr_score, accuracy_score, f1_score
from utils import load_best_model


def os_train_valid_model(root_save_path: str) -> None:
    """ Train & Valid Model using overall stock data.

    :param root_save_path: path to save the model

    """

    # ---- Some preparation ---- #
    os_model_save_path = f"{root_save_path}/model"
    os.makedirs(os_model_save_path, exist_ok=True)

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}  *****************")

    # ---- Make the dataset and dataloader ---- #
    train_dataset = FactoDataset(root_path=config.FACTOR_DATA_PATH, time_steps=config.TIME_STEPS, data_type="train")
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)  # the train dataloader
    valid_dataset = FactoDataset(root_path=config.FACTOR_DATA_PATH, time_steps=config.TIME_STEPS, data_type="valid")
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    if config.MODEL == "MLP":
        model = MLP_Net(input_size=config.FACTOR_NUM * config.TIME_STEPS, out_size=1, device=device)
    elif config.MODEL == "Big_MLP":
        model = Big_MLP_Net(input_size=config.FACTOR_NUM * config.TIME_STEPS, out_size=1, device=device)
    elif config.MODEL == "Conv":
        model = Conv_Net(device=device)
    elif config.MODEL == "GRU":
        model = GRU_Net(input_size=config.FACTOR_NUM, out_size=1, device=device)
    elif config.MODEL == "LSTM":
        model = LSTM_Net(input_size=config.FACTOR_NUM, out_size=1, device=device)
    elif config.MODEL == "Transformer":
        model = Transformer_Net(d_feat=config.FACTOR_NUM, out_size=1, device=device)
    else:
        raise ValueError(config.MODEL)
    # the loss function
    criterion = MSE_Loss(reduction=config.LOSS_REDUCTION)
    # the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # ---- Start Train and Valid ---- #
    # init the metric dict of all epochs
    epoch_metric = {
        "train_loss": np.zeros(config.EPOCHS), "valid_loss": np.zeros(config.EPOCHS),
        "train_R2": np.zeros(config.EPOCHS), "train_CORR": np.zeros(config.EPOCHS),
        "valid_R2": np.zeros(config.EPOCHS), "valid_CORR": np.zeros(config.EPOCHS),
        "train_ACC": np.zeros(config.EPOCHS), "train_F1": np.zeros(config.EPOCHS),
        "valid_ACC": np.zeros(config.EPOCHS), "valid_F1": np.zeros(config.EPOCHS)
    }
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
            features, labels = batch_data["feature"].to(device=device), batch_data["label"].to(device=device)
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
            train_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
            train_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
            last_step = now_step
        # note the loss and metrics for one epoch of TRAINING
        epoch_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
        epoch_metric["train_R2"][epoch] = r2_score(y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy())
        epoch_metric["train_CORR"][epoch] = corr_score(y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy())
        epoch_metric["train_ACC"][epoch] = accuracy_score(y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy())
        epoch_metric["train_F1"][epoch] = f1_score(y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy())
        # - valid model
        last_step = 0
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                # move data to device
                features, labels = batch_data["feature"].to(device=device), batch_data["label"].to(device=device)
                # forward to compute outputs, different model have different loss
                preds = model(features)
                loss = criterion(y_true=labels, y_pred=preds)
                # note the loss of valid in one iter
                valid_loss_one_epoch.append(loss.item())
                # doc the result in one iter
                now_step = last_step + preds.shape[0]
                valid_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
                valid_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
                last_step = now_step
        # note the loss and all metrics for one epoch of VALID
        epoch_metric["valid_loss"][epoch] = np.mean(valid_loss_one_epoch)
        epoch_metric["valid_R2"][epoch] = r2_score(y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy())
        epoch_metric["valid_CORR"][epoch] = corr_score(y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy())
        epoch_metric["valid_ACC"][epoch] = accuracy_score(y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy())
        epoch_metric["valid_F1"][epoch] = f1_score(y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy())
        # save model&model_config and metrics
        torch.save(model, f"{os_model_save_path}/model_pytorch_epoch_{epoch}")
        # write metric log
        dt = datetime.now() - t_start
        logging.info(f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, "
                     f"{['%s:%.4f ' % (key, value[epoch]) for key, value in epoch_metric.items()]}")
    # save the metric
    pd.DataFrame(epoch_metric).to_csv(f"{os_model_save_path}/model_metric.csv")
    # draw figure of train and valid metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(3, 1, 1)
    plt.plot(epoch_metric["train_loss"], label="train loss", color="g")
    plt.plot(epoch_metric["valid_loss"], label="valid loss", color="b")
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(epoch_metric["train_R2"], label="train R2", color="g")
    plt.plot(epoch_metric["valid_R2"], label="valid R2", color="b")
    plt.legend()
    plt.subplot(3, 2, 4)
    plt.plot(epoch_metric["train_CORR"], label="train CORR", color="g")
    plt.plot(epoch_metric["valid_CORR"], label="valid CORR", color="b")
    plt.legend()
    plt.subplot(3, 2, 5)
    plt.plot(epoch_metric["train_ACC"], label="train ACC", color="g")
    plt.plot(epoch_metric["valid_ACC"], label="valid ACC", color="b")
    plt.legend()
    plt.subplot(3, 2, 6)
    plt.plot(epoch_metric["train_F1"], label="train F1", color="g")
    plt.plot(epoch_metric["valid_F1"], label="Valid F1", color="b")
    plt.legend()
    plt.savefig(f"{root_save_path}/training_steps.png", dpi=200, bbox_inches="tight")
    logging.info("***************** TRAINING OVER ! *****************")


def os_pred_model(root_save_path: str) -> None:
    """ Test Model using single stock data.

    :param root_save_path: path to save the model

    """

    # ---- Some basic setting ---- #
    os_model_save_path = f"{root_save_path}/model"

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}  *****************")

    # ---- Make the dataset and dataloader ---- #
    test_dataset = FactoDataset(root_path=config.FACTOR_DATA_PATH, time_steps=config.TIME_STEPS)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
    logging.info(f"Test dataset: length = {len(test_dataset)}")
    preds_overall_stock = torch.zeros(len(test_dataset)).to(device=device)
    labels_overall_stock = torch.zeros(len(test_dataset)).to(device=device)

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    model, model_path = load_best_model(os_model_save_path, "valid_F1")
    logging.info(f"Load best mode {model_path}")

    # ---- Start Pred ---- #
    last_step = 0
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            # move data to device
            features, labels = batch_data["feature"].to(device=device), batch_data["label"].to(device=device)
            # forward to compute outputs, different model have different loss
            preds = model(features)
            # doc the result in one iter
            now_step = last_step + preds.shape[0]
            preds_overall_stock[last_step:now_step] = preds[:, 0].detach()
            labels_overall_stock[last_step:now_step] = labels[:, 0].detach()
            last_step = now_step

    # ---- Return os result ---- #
    logging.info(
        f"{preds_overall_stock.shape[0]} samples: "
        f"R2={r2_score(y_true=labels_overall_stock.cpu().numpy(), y_pred=preds_overall_stock.cpu().numpy())}, "
        f"CORR={corr_score(y_true=labels_overall_stock.cpu().numpy(), y_pred=preds_overall_stock.cpu().numpy())},"
        f"ACC={accuracy_score(y_true=labels_overall_stock.cpu().numpy(), y_pred=preds_overall_stock.cpu().numpy())}, "
        f"F1={f1_score(y_true=labels_overall_stock.cpu().numpy(), y_pred=preds_overall_stock.cpu().numpy())}"
    )


def pred_for_each_stock(root_save_path: str) -> None:
    """ Pred for each stock.

    :param root_save_path: path to save the model

    """

    # ---- Some basic setting ---- #
    os_model_save_path = f"{root_save_path}/model"

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}  *****************")

    # ---- For loop to read ---- #
    final_pred_df = None
    stock_csv_list = sorted(os.listdir(f"{config.FACTOR_DATA_PATH}/test_csv"))
    for stock_csv in stock_csv_list:
        stock_df = pd.read_csv(f"{config.FACTOR_DATA_PATH}/test_csv/{stock_csv}")
        # build up data
        test_dataset = FactoDataset(
            root_path=config.FACTOR_DATA_PATH, time_steps=config.TIME_STEPS,
            stock_file_list=[stock_csv.replace(".csv", ".npz")],
            data_type="test"
        )
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
        logging.info(f"Test dataset: length = {len(test_dataset)}")
        preds_single_stock = torch.zeros(len(test_dataset)).to(device=device)
        labels_single_stock = torch.zeros(len(test_dataset)).to(device=device)
        # load model
        model, model_path = load_best_model(os_model_save_path, "valid_F1")
        logging.info(f"Load best mode {model_path}")
        # predict
        last_step = 0
        with torch.no_grad():
            for batch_data in tqdm(test_loader):
                # move data to device
                features, labels = batch_data["feature"].to(device=device), batch_data["label"].to(device=device)
                # forward to compute outputs, different model have different loss
                preds = model(features)
                # doc the result in one iter
                now_step = last_step + preds.shape[0]
                preds_single_stock[last_step:now_step] = preds[:, 0].detach()
                labels_single_stock[last_step:now_step] = labels[:, 0].detach()
                last_step = now_step
        preds_single_stock_array = preds_single_stock.cpu().numpy()
        stock_df["preds"] = preds_single_stock_array
        stock_df = stock_df[["Date", "preds"]].rename(columns={"preds": stock_csv[:-4]})
        if final_pred_df is None:
            final_pred_df = stock_df
        else:
            final_pred_df = pd.merge(final_pred_df, stock_df, on="Date", how="outer")
    print(final_pred_df.isnull().sum().sum())
    # final_pred_df.to_csv(f"20240102_20240529.csv", index=False)


if __name__ == "__main__":
    # ---- Prepare some environments for training and prediction ---- #
    # fix the random seed
    fix_random_seed(seed=config.RANDOM_SEED)
    # build up the PATH
    SAVE_PATH = f"exp_os_reg_split/rs_{config.RANDOM_SEED}"
    LOG_FILE = f"{SAVE_PATH}/log_file.log"
    # build up the save directory of the PATH
    os.makedirs(SAVE_PATH, exist_ok=True)
    # construct the train&valid log file
    logging.basicConfig(filename=LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # ---- Step 1. Train & Valid model ---- #
    os_train_valid_model(root_save_path=SAVE_PATH)

    # ---- Step 2. Pred model ---- #
    os_pred_model(root_save_path=SAVE_PATH)

    # ---- Step 3. Pred for each book ---- #
    # pred_for_each_stock(root_save_path=SAVE_PATH)
