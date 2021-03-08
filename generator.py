import numpy as np


class Generator:
    def __init__(self, N, p, s, k, sigma):
        self.N = N
        self.p = p
        self.s = s
        self.k = k
        self.sigma = sigma

    def generate(self):
        # generating sample
        z = np.random.normal(0, 1, (self.N, self.p))
        X = np.ones((self.N, self.p))
        X[:, 0] = z[:, 0]
        for i in range(1, self.p):
            X[:, i] = self.k * X[:, i - 1] + z[:, i]

        # noise
        epsilon = np.random.normal(0, self.sigma ** 2, (int(self.N), 1))

        # generating ground truth
        theta = np.zeros((self.p, 1))
        theta[0:self.s] = 1
        # Y = X @ theta + epsilon
        # return X, Y, theta
        # theta = np.random.normal(0, 1, (self.p, 1))
        # theta_abs = np.abs(theta)
        # threshold = np.quantile(theta_abs, 1 - self.s / self.p)
        # mask = theta_abs > threshold
        # theta = mask * theta
        Y = X @ theta + epsilon
        return X, Y, theta
