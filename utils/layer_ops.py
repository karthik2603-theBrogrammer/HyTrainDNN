import os
import torch
from typing import Dict

import torch.nn as nn

def create_layer(layer_config: Dict) -> nn.Module:
    layer_type = layer_config["layer_type"]
    if layer_type == "Linear":
        return nn.Linear(layer_config["input"], layer_config["output"])
    elif layer_type == "Conv2d":
        return nn.Conv2d(layer_config["input"], layer_config["output"], 
                        layer_config.get("kernel_size", 3))
    elif layer_type == "Embedding":
        return nn.Embedding(layer_config["input"], layer_config["output"])
    elif layer_type == "LayerNorm":
        return nn.LayerNorm(layer_config["input"])
    elif layer_type == "Dropout":
        return nn.Dropout(layer_config.get("rate", 0.1))
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


