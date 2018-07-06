import logging # 记录日志
import os

import numpy as np
import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643
import torch
import torch.optim as optim
from tqdm import tqdm # 显示进度条
from tensorboardX import SummaryWriter # 记录训练效果

import utils
from train import train
from evaluate import evaluate

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, runs_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        runs_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(runs_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    writer = SummaryWriter(runs_dir)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        # 存储evaluate/accuracy, evaluate/loss, train/accuracy, train/loss
        for tag, value in val_metrics.items():
            writer.add_scalar('evaluate/'+tag, value, epoch)
        for tag, value in train_metrics.items():
            writer.add_scalar('train/'+tag, value, epoch)

        # 存储参数和其grad的分布
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins='auto')
            writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
            writer.add_scalar(tag+'/grad/mean', value.grad.data.mean(), epoch)

        # Save train and val metrics in a json file in the model directory
        # train_json_path = os.path.join(runs_dir, str(epoch)+".metrics_train.json")
        # utils.save_dict_to_json(train_metrics, train_json_path)
        # val_json_path = os.path.join(runs_dir, str(epoch)+".metrics_val.json")
        # utils.save_dict_to_json(val_metrics, val_json_path)

 
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(runs_dir, "best.metrics_val.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict' : optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=runs_dir,
                                epoch = epoch)

