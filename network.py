import numpy as np
from networkx import *


class FullyConnectedNetwork:
    def __init__(self, m):
        self.m = m
        self.w = 1/m * np.ones((m, m))

    def generate(self):
        return self.w

class ErodoRenyi:
    def __init__(self, node=5, probability=0.4):
        self.node = node
        self.probability = probability

    def generate(self):
        G = erdos_renyi_graph(self.node, self.probability)
        adjacent_matrix = to_numpy_matrix(G)
        matrix = np.zeros((self.node, self.node))
        for i in G.edges:
            degree = max(G.degree(i[0]), G.degree(i[1]))
            G.add_edge(*i, weight=1/(degree + 1))
        adjacent_matrix = to_numpy_array(G)
        weighted_matrix = np.eye(self.node) - np.diag(sum(adjacent_matrix)) + adjacent_matrix
        return weighted_matrix
