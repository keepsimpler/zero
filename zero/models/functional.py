import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    conv2d function implemented using *unfold* function 
    """
    N, in_channels, h_in, w_in = input.shape
    out_channels, in_channels, *kernel_size = weight.shape
    h_out = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0]-1)-1) // stride[0]) + 1   
    w_out = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) // stride[1]) + 1

    # a 3-D tensor of shape (N, in_channels * kernel_size[0] * kernel_size[1],
    # H_out * W_out)
    input_unfolded = F.unfold(input, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
    # transpose to the tensor of shape (N, H_out * W_out,
    # in_channels * kernel_size[0] * kernel_size[1])
    input_unfolded = input_unfolded.transpose(1, 2)
    # flatten kernels of shape (out_channels, in_channels, kernel_size[0], kernel_size[1])
    # to shape (out_channels, in_channels * kernel_size[0] * kernel_size[1]),
    # then transpose to shape (in_channels * kernel_size[0] * kernel_size[1], out_channels)
    weight = weight.view(weight.size(0), -1).t()
    # matrix multiple to shape (N, H_out * W_out, out_channels)
    output = input_unfolded.matmul(weight)
    # transpose to (N, out_channels, H_out * W_out)
    output = output.transpose(1, 2)
    if bias is not None:
        # broadcasting element-wise add bias of shape (out_channels, 1)
        output = output + bias.view(-1, 1)
    # squeeze to shape (N, out_channels, H_out, W_out)
    output = output.view(input.shape[0], out_channels, h_out, w_out)
    return output

def link_conv2d(input, act_fn, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    conv2d with non-linear link function

    Args:
    act_fn (function): a non-linear element-wise activation function

    warning: bias has shape of (out_channels)
    """
    N, in_channels, h_in, w_in = input.shape
    out_channels, in_channels, *kernel_size = weight.shape
    h_out = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0]-1)-1) // stride[0]) + 1   
    w_out = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) // stride[1]) + 1

    # a 3-D tensor of shape (N, in_channels * kernel_size[0] * kernel_size[1],
    # H_out * W_out)
    input_unfolded = F.unfold(input, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
    # transpose to the tensor of shape (N, H_out * W_out,
    # in_channels * kernel_size[0] * kernel_size[1])
    input_unfolded = input_unfolded.transpose(1, 2)
    # add a new dimension at the end of the tensor to get shape of
    # (N, H_out * W_out, in_channels * kernel_size[0] * kernel_size[1], 1)
    input_unfolded = input_unfolded.unsqueeze(-1)
    # flatten kernels of shape (out_channels, in_channels, kernel_size[0], kernel_size[1])
    # to shape (out_channels, in_channels * kernel_size[0] * kernel_size[1]),
    # then transpose to shape (in_channels * kernel_size[0] * kernel_size[1], out_channels)
    weight = weight.view(weight.size(0), -1).t()
    # broadcasting element-wise multiple to shape 
    # (N, H_out * W_out, in_channels * kernel_size[0] * kernel_size[1], out_channels)
    output = input_unfolded * weight
    # perform link activation functions
    output = act_fn(output)
    # sum according to dimension [in_channels * kernel_size[0] * kernel_size[1]] to
    # shape (N, H_out * W_out, out_channels)
    output = torch.sum(output, -2)
    # transpose to (N, out_channels, H_out * W_out)
    output = output.transpose(1, 2)
    if bias is not None:
        # broadcasting element-wise add bias of shape (out_channels, 1)
        output = output + bias.view(-1, 1)
    # squeeze to shape (N, out_channels, H_out, W_out)
    output = output.view(input.shape[0], out_channels, h_out, w_out)
    return output

def lv1(input, weight, bias=None):
    residential = F.linear(input, weight, bias) * input
    #output = input + residential
    return residential

def adaptive_step(input, stepwise):
    r"""
    input * stepwise, element-wise function
    Shape:
        - input: :math:`(N, *, in\_features), `*`代表任意多的附加维度
        - stepwise: :math:`(in\_features)`
        -output: math:`(N, *, in\_features)

    Note:
        输出维度＝＝输入维度
    """
    output = input * stepwise
    return output

def link_fc(input, act_fn, weight, bias=None):
    r"""
    非线性在边上，而不是在点上，的函数。:math:`y = f(x * A^T + b)` 是element-wise函数

    Shape:
        - input: :math:`(N, *, in\_features), `*`代表任意多的附加维度
        - act_fn: 边上的非线性函数
        - weight: :math:`(out\_features, in\_features)`
        - bias: :math:`(in\_features)`
        - output: :math:`(N, *, out\_features)`
    
    Note:
        bias长度是:math:`(in\_features)`, 而不是:math:`(out\_features)`

    """
    input = input.unsqueeze(-1) # 在输入的最后增加1个维度

    output = input * weight.t() # broadcasting element-wise 乘法
    if bias is not None:
        bias = bias.unsqueeze(-1) # 在bias的最后增加1个维度
        output += bias # broadcasting element-wise 加法
    # 执行非线性操作 (element-wise)
    output = act_fn(output)
    # add the input
    input = input / weight.shape[0]
    dims = list(input.shape[:-1])
    dims.append(weight.shape[0])
    input = input.expand(dims)
    output = input + output
    # 对倒数第2个维度(in_features)求和(or 求平均)
    output = torch.mean(output, -2) # / 2# sum mean
    return output


# 测量函数
def accuracy_fn(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (Tensor) dimension batch_size x 10 - log softmax output of the model
        labels: (Tensor) dimension batch_size, where each element is a value in 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Returns: (float) accuracy in [0,1]
    """
    outputs_labels = outputs.max(dim=1)[1]
    #print(outputs_labels, labels)
    return sum(outputs_labels == labels).item() / len(labels)

