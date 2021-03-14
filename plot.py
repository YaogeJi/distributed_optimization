import matplotlib.pyplot as plt
import queue


def plot(q:queue.Queue, config:dict):
    """

    """
    def _preprocess(config_dict:dict):
        param_dict = {}
        for subvalue in config_dict.values():
            if "param" in subvalue.keys():
                param_dict.update(subvalue["param"])
        param_dict.update({"model": config_dict["solver"]["model"]})
        return param_dict

    exp_info = []
    theta_hat = []
    loss_list = []
    x_axis = config['plot']['x_axis']
    y_axis = config['plot']['y_axis']
    type_loss = config['solver']['loss']
    stack = config['plot']['stack']
    parameter = config['plot']['parameter']
    param_dict = _preprocess(config)

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
    if x_axis == "constraint_param":
        x_label = r"$\lambda$"
    elif x_axis == "step_size":
        x_label = r"$\gamma$"
    else:
        x_label = x_axis
    if y_axis == "loss":
        if type_loss == "statistic_log_loss":
            if x_axis == 'iteration':
                y_label = r"$\log{\frac{1}{m}\sum||\theta_i-\theta^*||^2_2}$"
            else:
                y_label = r"$\log{\frac{1}{m}\sum||\hat{\theta}_i-\theta^*||^2_2}$"
        elif type_loss == "optimization_log_loss":
            if x_axis == 'iteration':
                y_label = r"$\log{\frac{1}{m}\sum||\theta_i-\hat{\theta}||^2_2}$"
            else:
                y_label = r"$\log{\frac{1}{m}\sum||\hat{\theta_i}-\hat{\theta}||^2_2}$"
        else:
            y_label = y_axis
    else:
        y_label = y_axis
    title = ''
    for seg in parameter:
        if seg != x_axis:
            if seg == 'constraint_param':
                title += r'$\lambda$' + '=' + str(param_dict[seg]) + ' '
            else:
                title += seg + '=' + str(param_dict[seg]) + ' '
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()