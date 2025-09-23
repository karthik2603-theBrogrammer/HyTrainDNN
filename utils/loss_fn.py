import torch
import torch.nn as nn

def LossFn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits, labels)