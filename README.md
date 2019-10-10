## Relation between Neural Networks and Complex Networks

The simplest definition of a neural network, more properly referred to as an 'artificial' neural network (ANN), is provided by the inventor of one of the first neurocomputers, Dr. Robert Hecht-Nielsen. He defines a neural network as:

> ... a computing **system** made up of a number of simple, **highly interconnected processing elements**, which process information by their dynamic state response to external inputs.

A neural network can be seen as a networked system.
And the theory of complex networks happened to be for the modeling of networked complex systems.

The definition of complex network on Wikipedia is:

>... a complex network is a **graph (network) with non-trivial topological features**—features that do not occur in simple networks such as lattices or random graphs but often occur in **graphs modeling of real systems**.

Therefore, neural networks are instances of networked complex systems that can be modeled using the theory of complex networks.


We try to introduce recent results of complex networks into neural networks, and want to evaluate a hyperthesis:

> The structural features of neural networks strongly influence the performance of them, and there exist several optimized structural features with better performance. 


## Installation

You can install Zero with pip: `pip install git+https://github.com/keepsimpler/zero`. 
Or clone this repo, cd to its directory, and `pip install -e .` .

Zero requires the latest stable [pytorch](http://www.pytorch.org) framework and [fastai](fast.ai) framework.
