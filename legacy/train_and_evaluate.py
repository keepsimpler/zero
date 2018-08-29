import logging # 记录日志
import os
from tqdm import tqdm # 显示进度条

import pandas as pd
import numpy as np
import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643
import torch
import torch.optim as optim

import utils
from train import train
from evaluate import evaluate

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, accuracy_fn, params, runs_dir):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        accuracy_fn: a function that computes accuracy using the output and labels of each batch
        params: (Params) hyperparameters
        runs_dir: (string) directory containing config, weights and log
    """

    best_val_acc = 0.0
    stats = pd.DataFrame()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # [params] go through functions below and and new statistic.

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, accuracy_fn, params)

        # Evaluate for one epoch on validation set
        evaluate(model, loss_fn, val_dataloader, accuracy_fn, params)

        params['epoch'] = epoch
        # 存储参数和其grad的分布
        for tag, value in model.named_parameters():
            params['parameter_stats_name'] = tag+'.norm'
            params['parameter_stats_value'] = value.data.norm().cpu().item()
            stats = stats.append(pd.DataFrame([params]), ignore_index=True)
            params['parameter_stats_name'] = tag+'.grad.norm'
            params['parameter_stats_value'] = value.grad.data.norm().cpu().item()
            stats = stats.append(pd.DataFrame([params]), ignore_index=True)
            #params[tag+'.norm'] = value.data.norm().cpu().item()
            #params[tag+'.grad.norm'] = value.grad.data.norm().cpu().item()
            #print(len(value[abs(value) <= 1e-5]), len(value))
 
        #stats = stats.append(pd.DataFrame([params]), ignore_index=True)
        #print(params)

        val_acc = params.evaluate_accuracy_avg
        is_best = val_acc>=best_val_acc
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict' : optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=runs_dir,
                                epoch = epoch)
    
    # stats.to_csv(os.path.join(runs_dir, 'stats.csv'), index=False)
    return stats
