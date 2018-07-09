import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .functional import link

class Link(Module):
    r"""非线性在边上，而不是在点上，的layer。:math:`y = f(x * A^T + b)` 是element-wise函数

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
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = Link(link, 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, act_fn, in_features, out_features, bias=True):
        super(Link, self).__init__()
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
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):        
        return link(input, self.act_fn, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class LinkContainer(nn.Module):
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
        super(LinkContainer, self).__init__()
        wide_of_layers = params.wide_of_layers # 每层的节点数
        # 参数中是由'-'分割的层节点数，例如，'784-30-10'
        # 这里要转换为整数列表，例如，[784,30,10]
        wide_of_layers = [int(x) for x in wide_of_layers.split('-')]
        act_fn_name = params.act_fn_name # 激活函数名称，函数来自于torch.nn.functional
        self.act_fn = getattr(F, act_fn_name) # 根据激活函数名称获得激活函数实例

        self.layers = nn.ModuleList(
            [Link(self.act_fn, wide_of_layers[i], wide_of_layers[i + 1]) 
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
            s = layer(s)
            #lin_s.retain_grad()
            # s = self.act_fn(lin_s) # F.relu

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)

