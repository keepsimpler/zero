"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643
import torch
import utils
import net as net
import data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--runs_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --runs_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, accuracy_fn, params):
    """Evaluate the model on test data batch by batch.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        accuracy_fn: a function that computes accuracy using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current training loop and a running average object for loss
    loss_avg = utils.RunningAverage()
    accuracy_avg = utils.RunningAverage()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)
        accuracy = accuracy_fn(output_batch, labels_batch)

        # update the average loss
        loss_avg.update(loss.data.item())  # loss.data[0]
        accuracy_avg.update(accuracy)

    metrics_string = "accuracy={:05.3f},loss={:05.3f}".format(accuracy_avg(), loss_avg())
    logging.info("- Evaluate metrics: " + metrics_string)
    params.evaluate_accuracy_avg = accuracy_avg()
    params.evaluate_loss_avg = loss_avg()
    return params


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join('.', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.runs_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    accuracy_fn = net.accuracy_fn
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.runs_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, accuracy_fn, params)
    save_path = os.path.join(args.runs_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
