
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: ../complexnet.ipynb

from .imports import *
from .core import *
from .graph import *

class NodeOP(nn.Module):
    """
    The Operation of every inner node in the network.

    Parameters:
    -----------
    ni : number of input channels
    no : number of output channels
    nh : number of hidden channels
    Unit : the operation at the node
    kwargs : arguments into `Unit`
    """
    def __init__(self, ni:int, no:int, nh:int, Unit:nn.Module, **kwargs):
        super(NodeOP, self).__init__()
        self.unit = Unit(ni, no, nh, **kwargs)

    def forward(self, *inputs):
        sum_inputs = sum(inputs)
        out = self.unit(sum_inputs)
        return out


class NetworkOP(nn.Module):
    """
    The operations along a DAG network.

    Parameters:
    -----------
    G   :  the `NetworkX` 'DiGraph' object, represent a DAG.
    ni  :  number of input channels of the network
    no  :  number of output channel of the network
    Unit : operation at every inner node
    stride : whether downsample or not, at the end of the network
    efficient : does using the checkpointing of `Pytorch` to save memory.
    kwargs : arguments into `Unit`

    """
    def __init__(self, G:nx.DiGraph, ni:int, no:int, Unit:nn.Module, stride:int=1,
                 efficient:bool=False, **kwargs):
        super(NetworkOP, self).__init__()
        self.efficient = efficient
        self.G = G
        self.n = G.graph['n'] # number of nodes
        self.nodeops = nn.ModuleList() # ! need improve, to Moduledict !
        for id in G.nodes(): # for each node
            if id == -1:  # if is the unique input node, do nothing
                continue
            elif id == self.n:  # if is the unique output node
                # then, concat its predecessors
                n_preds = len([*G.predecessors(id)])
                self.nodeops += [IdentityMapping(n_preds * ni, no, stride=stride)]
            else:  # if is the inner node
                self.nodeops += [NodeOP(ni, ni, ni, Unit, **kwargs)]

    def forward(self, x):
        results = {}
        results[-1] = x  # input data is the result of the unique input node
        for id in self.G.nodes(): # for each node
            if id == -1:  # if is the input node, do nothing
                continue
            # get the results of all predecessors
            inputs = [results[pred]  for pred in self.G.predecessors(id)]
            if id == self.n: # if is the output node
                cat_inputs = torch.cat(inputs, dim=1) # concat results of all predecessors
                if self.efficient:
                    return cp.checkpoint(self.nodeops[id], cat_inputs)
                else:
                    return self.nodeops[id](cat_inputs)
            else: # if is inner nodes
                if self.efficient:
                    results[id] = cp.checkpoint(self.nodeops[id], *inputs)
                else:
                    results[id] = self.nodeops[id](*inputs)

            # 删除前驱结点result中，不再需要的result
            for pred in self.G.predecessors(id):  # 获得节点的所有前驱结点
                succs = list(self.G.successors(pred))  # 获得每个前驱结点的所有后继节点
                # 如果排名最后的后继节点是当前节点，说明该前驱结点的result不再被后续的节点需要，可以删除
                if max(succs) == id:
                    del results[pred]



class ComplexNet(nn.Sequential):
    """
    Neural Network based on complex network

    Parameters:
    -----------
    Gs  :  a list of `NetworkX DiGraph` objects.
    ns  :  number of channels of all stages.
    Stage : class of a network.
    Unit : class of a node.
    c_out : number of output channels of the whole neural network.
    efficient : does using the checkpointing of `Pytorch` to save memory.
    kwargs : additional args into `Unit` class
    """
    def __init__(self, Gs:list, ns:list, Stage:nn.Module, Unit:nn.Module, c_out:int=10,
                 efficient:bool=False, **kwargs):
        super(ComplexNet, self).__init__()
        stem = conv_bn(3, ns[0])
        network_ops = []
        for i in range(len(ns)-2):
            network_ops += [Stage(Gs[i], ns[i], ns[i+1], Unit, stride=2, efficient=efficient, **kwargs)]
        # the last stage has stride=1 at its end, i.e. no downsampling
        network_ops += [Stage(Gs[-1], ns[-2], ns[-1], Unit, stride=1, efficient=efficient, **kwargs)]

        classifier = Classifier(ns[-1], c_out)
        super().__init__(
            stem,
            *network_ops,
            classifier
        )
        init_cnn(self)

def to_multiple_partitions(G):
    """
    Transform DAGs to Multiple Partitioned Networks.

    Returns:
    --------
    llinks : list of links among partitions
    partitions : number of nodes in all partitions
    """
    #first, convert `NetworkX` graph object to `Numpy` matrix order by node IDs
    A = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes))
    hcur = 1 # horizontal cursor, initialized as 1
    vcur = 0 # vertical cursor, initialized as 0
    n = A.shape[0] # number of nodes
    llinks = []  # initialize list of links as empty list
    partitions = []  # initialize number of nodes in partitions
    while vcur < n:
        current_nodes = list(range(vcur, hcur)) # current nodes
        # links to current partition  A[:vcur, current_nodes]
        links = []
        for row in range(vcur):
            for col in current_nodes:
                if A[row, col] == 1:
                    links.append((row, col - vcur))
        if current_nodes != [0]: # if not the start partition
            llinks.append(links)
            partitions.append(len(current_nodes))

        # move on vertical cursor
        vcur += len(current_nodes)
        # move on horizontal pointer
        while hcur < n and all(A[vcur:, hcur]==0):
            hcur += 1
        #print(vcur, hcur)

    return llinks, partitions

class Partition(nn.Module):
    """
    The operation of one partition in the multiple partitioned networks.
    One partition includes several nodes, which can be executed parallelly.

    Parameters:
    -----------
    ni : number of input channels of ONE node.
    cur_nodes : number of nodes in current partition.
    links : links from the nodes of all previous partitions to the nodes of current partition.
    has_groups: does group the nodes?
    Unit : the operation at the node.
    kwargs : arguments into `Unit`
    """
    def __init__(self, ni:int, cur_nodes:int, links:list, has_groups:bool, Unit:nn.Module, **kwargs):
        super(Partition, self).__init__()
        self.ni, self.cur_nodes = ni, cur_nodes
        self.links = torch.tensor(links)
        if has_groups:
            groups = self.cur_nodes
            kwargs['groups'] = groups # add to `kwargs`
        self.op = Unit(ni * self.cur_nodes, ni * self.cur_nodes, ni * self.cur_nodes, **kwargs)

    def forward(self, x):
        # number of channels of input should be `number of nodes in all previous partitions * self.ni`
        # number of channels of output should be `self.cur_nodes * self.ni`
        # first construct
        N,C,H,W = x.size()
        y = x.new_zeros((N, self.cur_nodes * self.ni, H, W))
        for i in range(self.links.size(0)):
            y[:, self.links[i,1] * self.ni : (self.links[i,1]+1) * self.ni, :, :] += \
            x[:, self.links[i,0] * self.ni : (self.links[i,0]+1) * self.ni, :, :]
        out = self.op(y)
        out = torch.cat([x, out], dim=1)
        return out

class Stage(nn.Module):
    """
    The operations along a multiple partitioned network.

    Parameters:
    -----------
    G   :  the `NetworkX` 'DiGraph' object, represent a DAG.
    ni  :  number of input channels of one node for the network
    no  :  number of output channel of one node for the network
    Unit : operation at every inner node
    stride : whether downsample or not, at the end of the network
    efficient : does using the checkpointing of `Pytorch` to save memory.
    has_groups: does group the nodes?
    kwargs : arguments into `Unit`

    """
    def __init__(self, G:nx.DiGraph, ni:int, no:int, Unit:nn.Module, stride:int=1,
                 efficient:bool=False, has_groups:bool=True, **kwargs):
        super(Stage, self).__init__()
        self.efficient = efficient
        llinks, partitions = to_multiple_partitions(G)
        self.llinks, self.partitions = llinks, partitions

        self.nodeops = nn.ModuleList() # ! need improve, to Moduledict !
        for i, links in enumerate(llinks):
            if i == len(llinks) - 1:  # reach the end of list
                n_preds = sum(partitions[:-1]) + 1 # all the previous nodes
                self.nodeops += [IdentityMapping(n_preds * ni, no, stride=stride)]
            else:  # inner partition
                self.nodeops += [Partition(ni, partitions[i], links, has_groups, Unit, **kwargs)]

    def forward(self, x):
        for nodeop in self.nodeops:
            if self.efficient:
                x = cp.checkpoint(nodeop, x)
            else:
                x = nodeop(x)
        return x