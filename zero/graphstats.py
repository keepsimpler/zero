
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: ../graphstats.ipynb

from .imports import *
from .core import *
from .graph import *

def complete_dag(n:int):
    """
    Generate a complete directed acyclic graph, which corresponds to architecture of ResNet.
    """
    G = nx.DiGraph()
    nodes = list(range(n))
    for id in nodes:
        for succ in range(id+1, n):
            G.add_edge(id, succ)
    return G

def dual_dag(n:int, d:int=2):
    """
    Generate a DAG, which corresponds to architecture of ResNetX when t=2.
    """
    G = nx.DiGraph()
    nodes = list(range(n))
    for id in nodes[:-1]:
        G.add_edge(id, id+1) # the backbone chain
        G.add_edge(0, id+1)  # the input node (0) link to all other nodes
        dest = id + 1 + d
        while dest < n:
            G.add_edge(id, dest)
            dest += d

    return G

def triple_dag(n:int):
    """
    Generate a DAG, which corresponds to architecture of ResNetX when t=3.
    """
    G = nx.DiGraph()
    nodes = list(range(n))
    for id in nodes[:-1]:
        G.add_edge(id, id+1)  # the backbone chain
        G.add_edge(0, id+1)  # the input node (0) link to all other nodes
        if id % 4 == 0 or id % 4 == 2:
            dest = id + 1 + 4
            while dest < n:
                G.add_edge(id, dest)
                dest += 4
        elif id % 4 == 1 or id % 4 == 3:
            dest = id + 1 + 2
            while dest < n:
                G.add_edge(id, dest)
                dest += 2

    return G

def quad_dag(n:int):
    """
    Generate a quad DAG.
    """
    G = nx.DiGraph()
    nodes = list(range(n))
    for id in nodes[:-1]:
        G.add_edge(id, id+1)  # the backbone chain
        G.add_edge(0, id+1)  # the input node (0) link to all other nodes
        if id % 6 == 0 or id % 6 == 3:
            dest = id + 1 + 6
            while dest < n:
                G.add_edge(id, dest)
                dest += 6
        elif id % 6 == 1 or id % 6 == 5:
            odd_even = 0
            dest = id + 1 + 2
            while dest < n:
                G.add_edge(id, dest)
                odd_even += 1
                if odd_even % 2 == 1:
                    dest += 4
                elif odd_even % 2 == 0:
                    dest += 2
        elif id % 6 == 2 or id % 6 == 4:
            odd_even = 0
            dest = id + 1 + 4
            while dest < n:
                G.add_edge(id, dest)
                odd_even += 1
                if odd_even % 2 == 1:
                    dest += 2
                elif odd_even % 2 == 0:
                    dest += 4

    return G

def incoherent(G):
    mat = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))

    in_degree = np.array([item[1] for item in sorted(G.in_degree())])
    in_degree[0] = 1
    Gamma = np.diag(in_degree) - mat
    levels = in_degree @ np.linalg.inv(Gamma)

    levels = np.array(levels).squeeze()

    temp = np.multiply(np.subtract.outer(levels, levels.T).T, mat)

    temp2 = temp[temp != 0]

    avg = np.mean(temp2)

    std1 = np.sqrt(np.mean(np.array(temp2).squeeze() ** 2) - 1)

    std2 = np.std(temp2)
    return avg, std1, std2