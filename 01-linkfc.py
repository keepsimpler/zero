from __future__ import print_function
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.optim import SGD

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss

from zero.models import LinkFCContainer, LinearContainer 

def get_data_loaders(train_size, val_size, train_batch_size, val_batch_size):
    """generator some fake data for continuous regression, with one independent variable"""
    # training data
    train_x = np.linspace(-7, 10, train_size)[:, np.newaxis]
    noise = np.random.normal(0, 2, train_x.shape)
    train_y = np.square(train_x) - 5 + noise

    # val data
    val_x = np.linspace(-7, 10, val_size)[:, np.newaxis]
    noise = np.random.normal(0, 2, val_x.shape)
    val_y = np.square(val_x) - 5 + noise

    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    val_x = torch.from_numpy(val_x).float()
    val_y = torch.from_numpy(val_y).float()

    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    val_dataset = Data.TensorDataset(val_x, val_y)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=2)

    return train_loader, val_loader, val_x, val_y


def run(epochs, lr, momentum, log_interval):
    train_loader, val_loader, test_x, test_y = get_data_loaders(train_size=2000,val_size=200,
        train_batch_size=64,val_batch_size=64)
    # print(next(iter(train_loader)))

    # model
    wide_of_layers = [10 for i in range(8)]
    wide_of_layers.insert(0,1) # insert 1 at index 0
    wide_of_layers.append(1) # append 1 
    act_fn = torch.tanh #F.tanh
    model = LinkFCContainer(wide_of_layers=wide_of_layers, act_fn=act_fn, batch_normalization=False)
    #model = LinearContainer(wide_of_layers=wide_of_layers, act_fn=act_fn, batch_normalization=False)

    outputs = {}

    def save_output(name):
        def hook(module, input, output):
            outputs[name] = output
        return hook

    #
    for name, module in model.named_modules():
        if list(module.children()) == []:
            module.register_forward_hook(save_output(name))

    grad_outputs = {}

    def save_grad_output(name):
        def hook(module, grad_input, grad_output):
            grad_outputs[name] = grad_output
        return hook

    #
    for name, module in model.named_modules():
        if list(module.children()) == []:
            module.register_backward_hook(save_grad_output(name))

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.mse_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'mse': Loss(F.mse_loss)},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            for key, value in outputs.items():
                print(key, value.data.mean().cpu().item(), value.data.std().cpu().item(),
                value.data.norm().cpu().item(), value.data.norm(p=3).cpu().item())
            for key, value in grad_outputs.items():
                print(key, value[0].data.mean().cpu().item(), value[0].data.std().cpu().item(),
                value[0].data.norm().cpu().item(), value[0].data.norm(p=3).cpu().item())
            # for name, param in model.named_parameters():
            #     print(name, param.data.mean().cpu().item(), param.data.std().cpu().item(),
            #     param.data.norm().cpu().item(), param.data.norm(p=3).cpu().item(),
            #     param.data.round().mode()[0])
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_mse = metrics['mse']
        for key, value in outputs.items():
            print(key, value)
        for key, value in grad_outputs.items():
            print(key, value)
        print("Training Results - Epoch: {}  Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_mse = metrics['mse']
        print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))

    trainer.run(train_loader, max_epochs=epochs)

    # model.eval()    # set eval mode to fix moving_mean and moving_var
    # preds = model(test_x)
    # plt.figure(3)
    # plt.plot(test_x.data.numpy(), preds.data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    # plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    # plt.legend(loc='best')
    # plt.show()
    # plt.ioff()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(args.epochs, args.lr, args.momentum, args.log_interval)
