import numpy as np
from projection import euclidean_proj_l1ball as proj

class Solver:
    def __init__(self, max_iteration, step_size, terminate_condition):
        self.max_iteration = max_iteration
        self.step_size = step_size
        self.terminate_condition = terminate_condition

    def fit(self, X, Y, comparison):
        raise NotImplementedError

    def show_param(self):
        raise NotImplementedError

class Lasso(Solver):
    def __init__(self, max_iteration, step_size, terminate_condition, solver_type, constraint_param):
        super(Lasso, self).__init__(max_iteration, step_size, terminate_condition)
        self.solver_type = solver_type
        self.constraint_param = constraint_param
        if solver_type == 0:
            self.lmda = constraint_param
        else:
            self.r = constraint_param

    def fit(self, X, Y, comparison=None):
        # initialize parameters we need
        loss = []
        N, p = X.shape
        # initialize iterates
        theta = 0.5 * np.ones((p, 1))
        # calculate value we need to use.
        C_1 = X.T @ X
        C_2 = X.T @ Y
        # define gradient methods

        def _lagrangian(t, radius=10):
            t = t - self.step_size * (1 / N * (C_1 @ t - C_2))
            t = np.sign(t) * np.clip(np.abs(t) - self.step_size * self.lmda, 0, None)
            t = proj(t, radius)
            return t

        def _projected(t):
            raise NotImplementedError

        # iterates!
        for step in range(self.max_iteration):
            if comparison is not None:
                cur_loss = np.log(np.linalg.norm(theta - comparison, ord=2))
                loss.append(cur_loss)
            theta_last = theta.copy()
            if self.solver_type == 0:
                theta = _lagrangian(theta)
            else:
                theta = _projected(theta)
            if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
                print("Early convergence, I quit.")
                return theta, loss, 0
        else:
            print("Max iteration, I quit.")
            return theta, loss, 1

    def show_param(self):
        return [self.step_size, self.constraint_param]


class DistributedLasso(Lasso):
    def __init__(self, max_iteration, step_size, terminate_condition, solver_type, constraint_param, w):
        super(DistributedLasso, self).__init__(max_iteration, step_size, terminate_condition, solver_type, constraint_param)
        self.w = w
        self.m = self.w.shape[0]

    def fit(self, X, Y, comparison=None):
        loss = []
        # Initialize parameters we need
        N, p = X.shape
        n = int(N / self.m)
        # Initialize iterates
        theta = 0.5 * np.ones((self.m, p))
        # Block data
        x = []
        y = []
        for i in range(self.m):
            x.append(X[n * i:n * (i + 1), :])
            y.append(Y[n * i:n * (i + 1), :])
            D = []
            E = []
        # block value we need to use
        for i in range(self.m):
            D.append(x[i].T @ x[i])
            E.append(y[i].T @ x[i])

        # define gradient methods
        def _lagrangian(t, radius=10):
            con = self.w @ t
            for i in range(self.m):
                t[i] = con[i] - self.step_size / n * (-E[i] + t[i].T @ D[i])
                t[i] = np.sign(t[i]) * np.clip(np.abs(t[i]) - self.step_size * self.lmda, 0, None)
                # projection
                temp = np.expand_dims(t[i].copy(), axis=1)
                temp = proj(temp, radius)
                t[i] = temp.squeeze()
            return t

        def _projected(t):
            raise NotImplementedError
        # iterates!

        for step in range(self.max_iteration):
            if comparison is not None:
                loss.append(np.log(np.linalg.norm(theta - np.repeat(comparison.T, self.m, axis=0), ord=2) / np.sqrt(self.m)))
            theta_last = theta.copy()
            if self.solver_type == 0:
                theta = _lagrangian(theta)
            else:
                theta = _projected(theta)
            if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition * np.sqrt(self.m):
                print("Early convergence, I quit.")
                return theta, loss, 0
        else:
            print("Max iteration, I quit.")
            return theta, loss, 1


class LocalizedLasso(Lasso):
    def __init__(self, max_iteration, step_size, terminate_condition, solver_type, constraint_param, m):
        super(LocalizedLasso, self).__init__(max_iteration, step_size, terminate_condition, solver_type, constraint_param)
        self.m = m

    def fit(self, X, Y, comparison=None):
        loss = []
        N, p = X.shape
        n = int(N / self.m)
        theta = 0.5 * np.ones((self.m, p))
        x = []
        y = []
        for i in range(self.m):
            x.append(X[n * i:n * (i + 1), :])
            y.append(Y[n * i:n * (i + 1), :])
            D = []
            E = []
        for i in range(self.m):
            D.append(x[i].T @ x[i])
            E.append(y[i].T @ x[i])

        def _lagrangian(t, radius=10):
            for i in range(self.m):
                t[i] = t[i] - self.step_size * (t[i].T @ D[i].T).T / n + self.step_size * E[i].squeeze() / n
                t[i] = np.sign(t[i]) * np.clip(np.abs(t[i]) - self.step_size * self.lmda, 0, None)
                # projection
                temp = np.expand_dims(t[i].copy(), axis=1)
                temp = proj(temp, radius)
                t[i] = temp.squeeze()
            return t

        def _projected(t):
            raise NotImplementedError

        for step in range(self.max_iteration):
            if comparison is not None:
                loss.append(np.log(np.linalg.norm(theta - np.repeat(comparison.T, self.m, axis=0), ord=2) / np.sqrt(self.m)))
            theta_last = theta.copy()
            if self.solver_type == 0:
                theta = _lagrangian(theta)
            else:
                theta = _projected(theta)
            if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition * np.sqrt(self.m):
                print("Early convergence, I quit.")
                return theta, loss, 0
        else:
            print("Max iteration, I quit.")
            return theta, loss, 1
