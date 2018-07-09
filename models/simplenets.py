
"""
定义网络，误差函数 和 测量函数
Defines the neural network, loss function and accuracy function
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearContainer(nn.Module):
    """
    PyTorch中定义网络的标准方式。在 __init__ 函数中选择网络的各层。
    然后，在 forward 函数中依次将各层作用于输入数据。
    在每层的结束可以应用 torch.nn.functional 包中定义的激活函数。
    要保证上一层结束后的输出数据的维度符合下一层输入数据的维度要求（即层与层之间的接口）。
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains：
                wide_of_layers 
                act_fn_name
        """
        super(LinearContainer, self).__init__()
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
        for layer in self.layers:
            lin_s = layer(s)
            #lin_s.retain_grad()
            s = self.act_fn(lin_s) # F.relu

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)

