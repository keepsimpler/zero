"""训练的主程序Train the model"""

import argparse
import logging
import os

import numpy as np
import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643
import torch
import torch.optim as optim
from tqdm import tqdm # 显示进度条

import utils
import net as net # 从model子目录导入网络模型
import data_loader as data_loader # 从model子目录导入数据loader
from train_and_evaluate import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/MNIST', help="数据目录Directory containing the dataset")
parser.add_argument('--runs_dir', default='runs/base_model', help="运行时目录")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --runs_dir containing weights to reload before \
                    training")


if __name__ == '__main__':

    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join('.', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # 解析 Params


    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.runs_dir, 'train.log'))
 
    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # restore pretrained model here
    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = os.path.join(args.runs_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # fetch loss function and accuracy function
    loss_fn = net.loss_fn
    accuracy_fn = net.accuracy_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, accuracy_fn, params, args.runs_dir)
    # print(params.dict) 对象都是引用，只有一个副本！！
    