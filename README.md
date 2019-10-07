# Zero, NN-based ML implemented with Pytorch

## Project Structure

│── models \
│   │── functional.py \
│   └── link_fc.py \
│   └── link_conv.py \
│── data \
│   │── MNIST \
│   │── CIFAR10 \
│   └── Imagenet \
│── runs \
│   │── checkpoints \
│   └── logs \
│── notebooks \
│── 01-tutorial.py \
│── 02-tutorial2.py \
│── ...... \
└── README.md \

## 01 link full connected network
bias is false; mean after link act or sum after link act; weight distribution is [-1,1] rather than [-1/sqrt(n),-1/sqrt(n)]; residential

Resnet: from relu to tanh, from two convs/bns to one conv/bn, plus relu, conv-->bn-->relu-->+shortcut, relu->conv->bn->+shortcut, conv->bn->relu->conv->bn->+shortcut->relu

## normalization / regularization
arg min max ||x^l|| / ||x^{l-1}||
max w.r.t. batch samples, find the maximum |||x^l|| / ||x^{l-1}|| among the batch samples.
min w.r.t. weight w^l

## resnet, wide resnet, resnext, densenet, automatically architecture searching


## Requires
. pytorch>0.4 \
. pytorch-ignite \
. fastai 