import json
import copy
import pickle

from generator import *
from network import *
from experiment import *
from solver import *


class Experiment:
    def __init__(self, config):
        self.exp_list = []
        self.data = []
        self.param = []
        data_param = config['data']
        network_param = config['network']
        solver_param = config['solver']
        exp_param = config["experiment"]["exp"]
        all_param = {"data": data_param, "network": network_param, "solver": solver_param}
        if exp_param in data_param['param'].keys():
            exp_param_position = "data"
        elif exp_param in network_param['param'].keys():
            exp_param_position = "network"
        elif exp_param in solver_param['param'].keys():
            exp_param_position = "solver"
        else:
            raise KeyError('Unknown experiment exp variable')

        for model in all_param["solver"]["model"]:
            all_param_single_model = copy.deepcopy(all_param)
            all_param_single_model['solver']['model'] = model
            all_param_single_model['solver']['param']['step_size'] = all_param['solver']['param']['step_size'][model]
            if exp_param == 'step_size':
                exp_param_list = all_param[exp_param_position]['param'][exp_param][model]
            else:
                exp_param_list = all_param[exp_param_position]['param'][exp_param]
            for exp_value in exp_param_list:
                all_param_single_exp = copy.deepcopy(all_param_single_model)
                all_param_single_exp[exp_param_position]['param'][exp_param] = exp_value
                w = self._network(all_param_single_exp['network'])
                if model == 0:
                    self.exp_list.append(Lasso(**all_param_single_exp['solver']['param']))
                elif model == 1:
                    self.exp_list.append(DistributedLasso(**all_param_single_exp['solver']['param'], w=w))
                elif model == 2:
                    self.exp_list.append(LocalizedLasso(**all_param_single_exp['solver']['param'], m=w.shape[0]))
                self.data.append(self._data(all_param_single_exp['data']))
                self.param.append(all_param_single_exp)

    def _data(self, data_param):
        key = hash(tuple(data_param['param'].values()))
        filepath = "data/{}.data".format(key)
        try:
            X, Y, ground_truth = pickle.load(open(filepath, "rb"))
        except:
            if data_param['method'] == 0:
                gen = Generator(**data_param['param'])
            else:
                raise NotImplementedError("unknown data generating method")
            X, Y, ground_truth = gen.generate()
            pickle.dump([X, Y, ground_truth], open(filepath, "wb"))
        return X, Y, ground_truth

    def _network(self, network_param):
        network_type = network_param['method']
        network_param = network_param['param']
        key = hash(tuple(network_param.values()))
        filepath = "data/{}.network".format(key)
        try:
            w = pickle.load(open(filepath, "rb"))
        except:
            if network_type == 0:
                # FullyConnectedGraph
                net = FullyConnectedNetwork(**network_param)
                w = net.generate()
                pickle.dump(w, open(filepath, "wb"))
            else:
                raise NotImplementedError("Unknown network type")
        return w
