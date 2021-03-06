import json
import multiprocessing as mp
import matplotlib.pyplot as plt
import queue

from experiment import *
"""
multiprocessing management
"""


def exp(instance, exp_info, X, Y, ground_truth, loss, q, exp_no):
    if loss == 'optimization_log_loss':
        theta_hat, _, error_code = instance.fit(X, Y)
        theta_hat, loss_list, error_code = instance.fit(X, Y, comparison=theta_hat)
    elif loss == 'statistic_log_loss':
        theta_hat, loss_list, error_code = instance.fit(X, Y, comparison=ground_truth)
    else:
        raise NotImplementedError("Unimplemented loss function")
    print(loss_list[-1])
    q.put([exp_info, theta_hat, loss_list])


def plot(q:queue.Queue, config:dict):
    """

    """
    def _preprocess(config_dict:dict):
        param_dict = {}
        for subvalue in config_dict.values():
            param_dict.update(subvalue["param"])
        param_dict.update({"model": config_dict["solver"]["model"]})
        return param_dict

    exp_info = []
    theta_hat = []
    loss_list = []
    x_axis = config['plot']['x_axis']
    y_axis = config['plot']['y_axis']
    stack = config['plot']['stack']
    parameter = config['plot']['parameter']
    title = {}
    legend_var = {}
    x = {}
    y = {}

    while not q.empty():
        one_optim = q.get()
        exp_info.append(_preprocess(one_optim[0]))
        theta_hat.append(one_optim[1])
        loss_list.append(one_optim[2])

    for i, info in enumerate(exp_info):
        if x_axis != "iteration":
            if info[stack] not in x.keys():
                print(info[stack])
                x[info[stack]] = [info[x_axis]]
                y[info[stack]] = [loss_list[i][-1]]
            else:
                x[info[stack]].append(info[x_axis])
                y[info[stack]].append(loss_list[i][-1])
        else:
            x[i] = list(range(len(loss_list[i])))
            y[i] = loss_list[i]
    model_name = ["centralize", "distributed", "localized"]
    for this_stack, _x in x.items():
        print(this_stack)
        if stack == "model":
            legend = model_name[this_stack]
        else:
            legend = this_stack
        _y = y[this_stack]
        plt.plot(_x, _y, label=legend)
    plt.legend(loc='upper right')
    plt.show()


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
