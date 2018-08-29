# coding=utf-8
"""
functions, layers, and containers with nonlinear link function
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .functional import link_fc

class LinkFC(Module):
    r"""非线性在边上，而不是在点上，的full connected layer。
    :math:`y = f(x * A^T + b)` 是element-wise函数

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (in_features)

    Examples::

        >>> m = Link(link, 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """
    def __init__(self, act_fn, in_features, out_features, bias=False):
        super(LinkFC, self).__init__()
        self.act_fn = act_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """init weights and bias by a distribution"""
        stdv = 1. # / math.sqrt(self.weight.size(1)) # in_features 大小
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return link_fc(x, self.act_fn, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class LinkFCContainer(nn.Module):
    r"""
    Args:
        wide_of_layers (list): wide of layers
        act_fn (function): link activation function
        batch_normalization (bool): if do batch normalize. default is False
"""
    def __init__(self, wide_of_layers, act_fn, batch_normalization=False):
        super(LinkFCContainer, self).__init__()
        self.wide_of_layers = wide_of_layers
        self.act_fn = act_fn # getattr(F, act_fn) # 根据激活函数名称获得激活函数实例
        self.batch_normalization = batch_normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(wide_of_layers[i], momentum=0.5)
            for i in range(len(wide_of_layers) - 1)])

        self.fc_layers = nn.ModuleList([
            LinkFC(self.act_fn, wide_of_layers[i], wide_of_layers[i + 1])
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
        return x
