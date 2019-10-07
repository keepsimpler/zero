# coding=utf-8
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import lv1

class LV1(nn.Module):
    r"""
    LV1 model
    """

    def __init__(self, in_features, out_features, bias=False):
        super(LV1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1)) # in_features 大小
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return lv1(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class LV1Container(nn.Module):

    def __init__(self, params):
        """
        Args:
            params: (Params) contains：
                wide_of_layers 
        """
        super(LV1Container, self).__init__()
        wide_of_layers = params.wide_of_layers # 每层的节点数
        # 参数中是由'-'分割的层节点数，例如，'784-30-10'
        # 这里要转换为整数列表，例如，[784,30,10]
        wide_of_layers = [int(x) for x in wide_of_layers.split('-')]

        layers = nn.ModuleList()
        layers.append(nn.Linear(wide_of_layers[0], wide_of_layers[1]))
        layers.extend(
            nn.ModuleList(
                [LV1(wide_of_layers[i], wide_of_layers[i + 1]) 
                for i in range(1, len(wide_of_layers) - 2)
                ]
             )
        )
        layers.append(nn.Linear(wide_of_layers[-2], wide_of_layers[-1]))
        self.layers = layers

    def forward(self, s):
        """
        定义如何使用网络的各层和激活函数来作用于输入数据

        Args:
            s: (Tensor) contains a batch of images, of dimension batch_size x (1 * 28 * 28) .

        Returns:
            out: (Tensor) dimension batch_size x 10 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        for layer in self.layers:
            s = layer(s)
            #lin_s.retain_grad()
            # s = self.act_fn(lin_s) # F.relu

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)


class LinearContainer(nn.Module):
    r"""
    Args:
        wide_of_layers (list): wide of layers
        act_fn (function): link activation function
        batch_normalization (bool): if do batch normalize. default is False
"""
    def __init__(self, wide_of_layers, act_fn, batch_normalization=False):
        super(LinearContainer, self).__init__()
        self.wide_of_layers = wide_of_layers
        self.act_fn = act_fn # getattr(F, act_fn) # 根据激活函数名称获得激活函数实例
        self.batch_normalization = batch_normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(wide_of_layers[i], momentum=0.5)
            for i in range(len(wide_of_layers) - 1)])

        self.fc_layers = nn.ModuleList([
            nn.Linear(wide_of_layers[i], wide_of_layers[i + 1])
            for i in range(len(wide_of_layers) - 1)])

    def forward(self, x):
        r"""
        定义如何使用网络的各层和激活函数来作用于输入数据

        Args:
            x: (Tensor) contains a batch of samples,with shape of :math:`(N, *, in\_features)` .

        Returns:
            out: (Tensor) the log probabilities for the labels of a batch of samples,
            with shape of :math:`(N, # of classes)` .

        """
        for bn_layer, fc_layer in zip(self.bn_layers, self.fc_layers):
            if self.batch_normalization:
                x = bn_layer(x)
            x = fc_layer(x)
            x = self.act_fn(x)
        return x

class LinearResContainer(nn.Module):

    def __init__(self, params):
        """
        Args:
            params: (Params) contains：
                wide_of_layers 
                act_fn_name
        """
        super(LinearResContainer, self).__init__()
        wide_of_layers = params.wide_of_layers # 每层的节点数
        # 参数中是由'-'分割的层节点数，例如，'784-30-10'
        # 这里要转换为整数列表，例如，[784,30,10]
        wide_of_layers = [int(x) for x in wide_of_layers.split('-')]
        act_fn_name = params.act_fn_name # 激活函数名称，函数来自于torch.nn.functional
        self.act_fn = getattr(F, act_fn_name) # 根据激活函数名称获得激活函数实例

        self.layers = nn.ModuleList(
            [nn.Linear(wide_of_layers[i], wide_of_layers[i + 1]) 
             for i in range(len(wide_of_layers) - 1)])

    def forward(self, s):
        """
        定义如何使用网络的各层和激活函数来作用于输入数据

        Args:
            s: (Tensor) contains a batch of images, of dimension batch_size x (1 * 28 * 28) .

        Returns:
            out: (Tensor) dimension batch_size x 10 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        for i, layer in enumerate(self.layers):
            lin_s = layer(s)
            #lin_s.retain_grad()
            residential = self.act_fn(lin_s) # F.relu
            if i > 0 and i < len(self.layers)-1:
                s = s + residential
            else:
                s = residential


        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)

