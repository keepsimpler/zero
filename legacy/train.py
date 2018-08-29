import logging
from tqdm import tqdm # 显示进度条

import numpy as np
import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643
import torch
import torch.optim as optim

import utils

def train(model, optimizer, loss_fn, dataloader, accuracy_fn, params):
    """Train the model on train data batch-by-batch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        accuracy_fn: a function that computes accuracy using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    loss_avg = utils.RunningAverage()
    accuracy_avg = utils.RunningAverage()

    # Use tqdm for progress bar
#    with tqdm(total=len(dataloader)) as t:
    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        accuracy = accuracy_fn(output_batch, labels_batch)

        #for tag, value in model.named_parameters():
        #    logging.info(str(len(value[abs(value) <= 1e-10])) + str(len(value)))

        # clear previous gradients, compute gradients of all tensors wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.data.item())  # loss.data[0]
        accuracy_avg.update(accuracy)
#            t.set_postfix(loss='{:05.3f}'.format(loss_avg()),
#            accuracy='{:05.3f}'.format(accuracy_avg()))
#            t.update()

    metrics_string = "accuracy={:05.3f},loss={:05.3f}".format(accuracy_avg(), loss_avg())
    logging.info("- Train metrics: " + metrics_string)
    params['train_accuracy_avg'] = accuracy_avg()
    params['train_loss_avg'] = loss_avg()
    return params
