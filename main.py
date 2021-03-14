import json
import multiprocessing as mp
import matplotlib.pyplot as plt
import queue

from plot import plot
from experiment import *
"""
multiprocessing management
"""


def exp(instance, exp_info, X, Y, ground_truth, loss, q, exp_no):
    if loss == 'optimization_log_loss':
        theta_hat, _, error_code = instance.fit(X, Y, ground_truth)
        theta_hat, loss_list, error_code = instance.fit(X, Y, ground_truth, comparison=theta_hat)
    elif loss == 'statistic_log_loss':
        theta_hat, loss_list, error_code = instance.fit(X, Y, ground_truth, comparison=ground_truth)
    else:
        raise NotImplementedError("Unimplemented loss function")
    print(loss_list[-1])
    q.put([exp_info, theta_hat, loss_list])


def main():
    with open("config.json") as f:
        config = json.load(f)
    experiment = Experiment(config)
    exp_list = experiment.exp_list
    exp_info = experiment.param
    # start processing
    loss = config['solver']['loss']
    is_mp = config["multiprocessing"]["valid"]
    if is_mp:
        max_proc = config["multiprocessing"]["max_proc"]
        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(processes=max_proc)
        for count, inst in enumerate(exp_list):
            X, Y, ground_truth = experiment.data[count]
            pool.apply_async(exp, args=(inst, exp_info[count], X, Y, ground_truth, loss, q, count))
        pool.close()
        pool.join()
    else:
        q = queue.Queue()
        for count, inst in enumerate(exp_list):
            X, Y, ground_truth = experiment.data[count]
            exp(inst, exp_info[count], X, Y, ground_truth, loss, q, count)
    plot(q, config)


if __name__ == "__main__":
    main()
