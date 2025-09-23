# Simple DNN
dnn_schema = [
    {
        "layer_number": 1, 
        "input": 512, 
        "output": 768, 
        "layer_type": "Linear", 
        "layer_name": "DNN_Linear_1"
    },
    {
        "layer_number": 2, 
        "input": 768, 
        "output": 1536, 
        "layer_type": "Linear", 
        "layer_name": "DNN_Linear_2"
    },
    {
        "layer_number": 3, 
        "input": 1536, 
        "output": 10, 
        "layer_type": "Linear", 
        "layer_name": "DNN_Linear_3"
    }
]

# CNN for images
cnn_schema = [
    {"layer_number": 1, "input": 3, "output": 64, "kernel_size": 3, "layer_type": "Conv2d", "layer_name": "conv1"},
    {"layer_number": 2, "input": 64, "output": 128, "kernel_size": 3, "layer_type": "Conv2d", "layer_name": "conv2"},
    {"layer_number": 3, "input": 128*56*56, "output": 256, "layer_type": "Linear", "layer_name": "fc1"},
    {"layer_number": 4, "input": 256, "output": 10, "layer_type": "Linear", "layer_name": "classifier"}
]

# Transformer block
transformer_schema = [
    {"layer_number": 1, "input": 50000, "output": 512, "layer_type": "Embedding", "layer_name": "embedding"},
    {"layer_number": 2, "input": 512, "layer_type": "LayerNorm", "layer_name": "ln1"},
    {"layer_number": 3, "input": 512, "output": 2048, "layer_type": "Linear", "layer_name": "ffn1"},
    {"layer_number": 4, "rate": 0.1, "layer_type": "Dropout", "layer_name": "dropout1"},
    {"layer_number": 5, "input": 2048, "output": 512, "layer_type": "Linear", "layer_name": "ffn2"},
    {"layer_number": 6, "input": 512, "layer_type": "LayerNorm", "layer_name": "ln2"}
]

# 10B parameter model (simplified)
large_model_schema = [
    {"layer_number": 1, "input": 50000, "output": 4096, "layer_type": "Embedding", "layer_name": "embedding"},
    {"layer_number": 2, "input": 4096, "output": 16384, "layer_type": "Linear", "layer_name": "ffn1"},
    {"layer_number": 3, "input": 16384, "output": 4096, "layer_type": "Linear", "layer_name": "ffn2"},
    {"layer_number": 4, "input": 4096, "output": 50000, "layer_type": "Linear", "layer_name": "lm_head"}
]