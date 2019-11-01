from collections import OrderedDict

from fastai.vision import *

import torch.utils.checkpoint as cp # checkpointing

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

# import pytorch_lightning as pl

import numpy as np
from numpy.linalg import matrix_power # for calculation of paths
import networkx as nx

import matplotlib.pylab as plt