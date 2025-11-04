import torch 
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config
from transformers.models.llama.configuration_llama import LlamaConfig

import time 

MODEL_CONFIGS = {
    "100m": LlamaConfig(
        max_position_embeddings=4096,
        num_hidden_layers=4,
        num_attention_heads=32,
        intermediate_size=2048,
        hidden_size=1024,
    ),
    "3b": LlamaConfig(max_position_embeddings=4096, num_hidden_layers=20, num_attention_heads=32, intermediate_size=2048, hidden_size=5120),
    "5b": LlamaConfig(max_position_embeddings=4096, num_key_value_heads=8),
    "7b": LlamaConfig(max_position_embeddings=4096),
    "10b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=30,
        num_attention_heads=40,
        max_position_embeddings=4096,
    ),
    "13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=4096,
    ),
    "70b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=4096,
        num_key_value_heads=8,
    ),
}

