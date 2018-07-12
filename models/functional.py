import torch
import torch.nn.functional as F

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

def link(input, act_fn, weight, bias=None):
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
    # 对倒数第2个维度(in_features)求和
    output = torch.sum(output, -2)
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
    return sum(outputs_labels == labels).item() / len(labels)

