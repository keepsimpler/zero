# coding=utf-8
## copied from https://stackoverflow.com/questions/51459987/variation-between-custom-convolution-vs-pytorch-conv2d-results

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.utils import _pair
from functional import custom_conv2d, link_conv2d

class CustomConv2d(nn.Module):

    r"""
    Customized convolution layer

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor 
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size[0], kernel_size[1]).
            The values of these weights are sampled from uniform distribution
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{\text{in\_channels} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor): the learnable bias of the model of shape (out_channels).
            If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{\text{in\_channels} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(CustomConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0],
        self.kernel_size[1]))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = link_conv2d(input, F.tanh, self.weight, self.bias, self.stride, self.padding, self.dilation)
        #output = custom_conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation)

        return output


if __name__ == '__main__':

    torch.manual_seed(1)

    input = torch.randn(1,2,3,3)

    conv1 = nn.Conv2d(input.shape[1], out_channels=3, kernel_size=2, dilation=1, padding=1, stride=1, bias = True)
    conv1_output = conv1(input)

    torch.manual_seed(1)

    conv2 = CustomConv2d(in_channels=input.shape[1], out_channels=3, kernel_size=2,  dilation=1, stride =1, padding = 1, bias = True)

    conv2.weight = conv1.weight
    conv2.bias = conv1.bias

    conv2_output = conv2(input)

    print(torch.equal(conv1.weight.data, conv2.weight.data))

    print(torch.equal(conv1_output, conv2_output))