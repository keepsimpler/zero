"""训练的主程序Train the model"""

# 导入Python公共包
import argparse
import logging
import os
from tqdm import tqdm # 显示进度条

# 导入数据处理和人工智能相关的包
import pandas as pd
import numpy as np
import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 导入自己写的包
import utils
import models # 从models子目录导入网络模型
import models.functional as F2 # 自定义的函数
import data_loader as data_loader # 导入数据loader
from train_and_evaluate import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--params_file', default='params.csv', help="当前目录下的参数文件的名字")
#parser.add_argument('--restart', default='True', help="是否重新开始本模型的训练")
parser.add_argument('--restore_file', default=None, help="可选，已训练模型文件的名字，在runs目录下")


if __name__ == '__main__':

    args = parser.parse_args()
    runs_dir = "runs/"
    # Set the logger
    utils.set_logger(os.path.join(runs_dir, 'train.log'))
    # statistic file
    stats_file = args.params_file.split('.')[0] + '_stats.csv'

    # 获得参数
    df = pd.read_csv(os.path.join(runs_dir, args.params_file))
    for index, params in df.iterrows():
        print(params)
        # Use GPU if available
        # and Set the random seed for reproducible experiments
        params.cuda = torch.cuda.is_available()
        if params.cuda:
            torch.cuda.set_device(params.cuda_device)
            torch.cuda.manual_seed(230)
        else:
            torch.manual_seed(230)

        # Create the input data pipeline
        logging.info("Loading the datasets...")

        # fetch dataloaders
        data_dir = os.path.join('data/', params.dataset_name)
        dataloaders = data_loader.fetch_dataloader(['train', 'val'], data_dir, params.dataset_name, params)
        train_dl = dataloaders['train']
        val_dl = dataloaders['val']

        logging.info("- done.")

        model_name = params.model_name # 模型名称，模型来自于models包
        model_cls = getattr(models, model_name) # 根据模型名称获得模型类
        # 根据模型类生成模型实例
        model = model_cls(params).cuda() if params.cuda else model_cls(params)
        #if params.cuda: # torch.cuda.device_count() > 1
        #    logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        #    model = nn.DataParallel(model, device_ids=[0,1,2,3])
        #    model.cuda()
        #    model.to(device)
        #model.apply(F2.init_weights)

        if params.optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
        elif params.optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        elif params.optimizer_name == 'SparseAdam':
            optimizer = optim.SparseAdam(model.parameters(), lr=params.learning_rate)
        else:
            logging.error("- optimizer does not exist.")

        # fetch loss function and accuracy function
        loss_fn = F.nll_loss # nn.NLLLoss().cuda()
        accuracy_fn = F2.accuracy_fn

        # restore pretrained model here
        # reload weights from restore_file if specified
        if args.restore_file is not None:
            restore_path = os.path.join(runs_dir, args.restore_file + '.pth.tar')
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        stats = train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, accuracy_fn, params, runs_dir)

        if index == 0:
            with open(os.path.join(runs_dir, stats_file), 'w') as f:
                stats.to_csv(f, index=False)
        else:
            with open(os.path.join(runs_dir, stats_file), 'a') as f:
                stats.to_csv(f, index=False, header=False)

        # print(params) Python对象都是引用，只有一个副本！！
