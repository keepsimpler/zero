
import numpy as np

import cv2 # 为了能正确导入torch,见 https://github.com/pytorch/pytorch/issues/643

from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# define a training image loader that specifies transforms on images.
train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
     # x.view(-1, np.prod(x.size()[-3:])) 合并(C,H,W)3个维度 为了使用简单网络
    transforms.Lambda(lambda x: x.view(-1))
])

# loader for evaluation, same with train
eval_transformer = train_transformer

def fetch_dataloader(types, data_dir, params):
    """
    获得DataLoader对象
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters 超参数 对象
                including: train_batch_size, 
                           test_batch_size
                           num_workers

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=train_transformer)
    eval_ds = datasets.MNIST(data_dir, train=False, download=True, transform=eval_transformer)
    for split in ['train', 'val', 'test']:
        if split in types:

            if split == 'train':
                dl = DataLoader(train_ds,
                    batch_size=params.train_batch_size, shuffle=True, 
                    num_workers=params.num_workers)
            else: # train flag is False
                dl = DataLoader(eval_ds,
                    batch_size=params.train_batch_size, shuffle=True, 
                    num_workers=params.num_workers)

            dataloaders[split] = dl

    return dataloaders