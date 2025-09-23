import torch
import torch.nn as nn
import os
from typing import Dict, List

from .layer_ops import create_layer

def prepare_checkpoints(config: List[Dict], checkpoint_path: str = "model_checkpoints"):
    """Prepare checkpoints for each layer in the configuration."""
    os.makedirs(checkpoint_path, exist_ok=True)
    
    for layer_config in config:
        layer = create_layer(layer_config)
        layer_name = layer_config["layer_name"]
        
        checkpoint_file = os.path.join(checkpoint_path, f"{layer_name}.pt")
        torch.save({
            'state_dict': layer.state_dict(),
            'config': layer_config
        }, checkpoint_file)
        
        print(f"Saved checkpoint: {checkpoint_file}")
    
    # Return the checkpoint path so it can be used by the Trainer
    return checkpoint_path

def load_checkpoint(layer_name: str, checkpoint_path: str = "model_checkpoints") -> tuple:
    """Load a checkpoint for a specific layer."""
    checkpoint_file = os.path.join(checkpoint_path, f"{layer_name}.pt")
    checkpoint = torch.load(checkpoint_file)
    
    layer = create_layer(checkpoint['config'])
    layer.load_state_dict(checkpoint['state_dict'])
    
    return layer, checkpoint['config']