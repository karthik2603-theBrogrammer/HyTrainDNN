import torch 
import numpy as np

class SkipParEpochManager(object):
    def __init__(self, num_epochs, num_layers, num_steps, filter_percentile = 0.001):
        self.epochs = num_epochs
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.current_layer_norm_map = {}
        self.layerid_norm_map = {}
        self.layerid_normlist_map = {}
        self.layer_eta_map = {}
        self.stale_array = None
        self.skip_window = None
        self.filter_percentile = filter_percentile
        self.interval_layer_norm_maps = []
        self.intervals = []

    def init_layer_id_norm_map(self):
        for layer in range(1, self.num_layers+1):
            self.layerid_norm_map[layer] = 0.0
            self.layerid_normlist_map[layer] = []
        self.current_layer_norm_map = {}
        return None
    
    def clear_layerid_normlist_map(self):
        for layer in range(1, self.num_layers+1):
            self.layerid_normlist_map[layer] = []
        return None


    def add_norms_to_layermap(self, layer_id, norm):
        self.layerid_normlist_map[layer_id].append(norm)
        return None
 
    # helper functions to perform embedded epoch operations. 
    def get_weighted_avg_of_norms(self, weights):
        for layer_id in range(1, self.num_layers+1):
            norm_list = self.layerid_normlist_map[layer_id]
            grad_norm = np.average(norm_list, weights = weights[layer_id])
            self.current_layer_norm_map[layer_id] = grad_norm
        return None

    def accumulate_layer_norm_map(self):
        for layer in range(1, self.num_layers+1):
            self.layerid_norm_map[layer] += self.current_layer_norm_map[layer]

    def add_interval_layer_norm_map(self, step, layer_norm_map):
        print((step, layer_norm_map))
        self.interval_layer_norm_maps.append((step, layer_norm_map))


    def calculate_etas(self):

        # use autofreeze logic to find the relative change of norms in 
        # each layer of the model to check convergence.
        etas = []
        step_t1, norm_del_t1 = self.interval_layer_norm_maps[-1]
        step_t2, norm_del_t2 = self.interval_layer_norm_maps[-2]


        for layer in range(1, self.num_layers+1):
            a = norm_del_t1[layer]/ step_t1
            b = norm_del_t2[layer]/ step_t2
            eta = abs(b - a)/ b
            self.layer_eta_map[layer] = eta
            etas.append((layer, eta))
        print(f"=== etas: {etas}")
        return etas

    def calculate_stale_array(self, etas):
        # eta's are estimated from `calculate_etas` function
        # Filter out NaN/inf values before sorting to get a valid percentile
        valid_etas = [(layer, eta) for layer, eta in etas if not (np.isnan(eta) or np.isinf(eta))]
        valid_etas.sort(key=lambda x: x[1])
        
        if not valid_etas:
            print("All Eta values are not valid.")
            self.stale_array = [1 for l_id in range(1, self.num_layers + 1)]
            return 
        
        num_layers_to_filter = int(self.filter_percentile * self.num_layers)
        kth_percentile_eta = valid_etas[int(self.filter_percentile * len(valid_etas))][1]
        below_percentile_layers = [layer for layer, eta in valid_etas if eta <= kth_percentile_eta]

        nan_layers = [layer for layer, eta in etas if np.isnan(eta) or np.isinf(eta)]
        layers_to_filter = set(below_percentile_layers[:num_layers_to_filter])

        if len(layers_to_filter) < num_layers_to_filter:
            remaining_needed = num_layers_to_filter - len(layers_to_filter)
            layers_to_filter.update(nan_layers[:remaining_needed])

        self.stale_array = [1 if l_id in layers_to_filter else 0 for l_id in range(1, self.num_layers + 1)]
        print("=== Stale Array: ", self.stale_array)
