import numpy as np


class FullyConnectedNetwork:
    def __init__(self, m):
        self.m = m
        self.w = 1/m * np.ones((m, m))

    def generate(self):
        return self.w