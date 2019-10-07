"""
PyTorch中定义网络的标准方式。在 __init__ 函数中选择网络的各层。
然后，在 forward 函数中依次将各层作用于输入数据。
在每层的结束可以应用 torch.nn.functional 包中定义的激活函数。
要保证上一层结束后的输出数据的维度符合下一层输入数据的维度要求（即层与层之间的接口）。
"""

from .simplenet import *
from .linknet import *
from .resnet import *

__all__ = [
    'LinearContainer',
    'LinearResContainer',
    'LV1',
    'LV1Container',
    'LinkFC',
    'LinkFCContainer',
    'functional',
    'ResNet18'
    'functional'
]
