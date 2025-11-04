def get_layers_in_model(model):
    # Supports llama2 and GPT2 models.

    def extract_layer_identifier(param_name):
        parts = param_name.split(".")
        
        # Handle Llama-style: model.layers.18.* -> model.layers.18.
        if 'layers' in parts:
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = i + 1
                    break
            if layer_idx is not None:
                return '.'.join(parts[:layer_idx + 1]) + '.'
        
        # Handle GPT-style: model.transformer.h.18.* -> model.transformer.h.18.
        elif 'h' in parts:
            h_idx = None
            for i, part in enumerate(parts):
                if part == 'h' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    h_idx = i + 1
                    break
            if h_idx is not None:
                return '.'.join(parts[:h_idx + 1]) + '.'
        
        # For non-transformer blocks (embeddings, final norms, etc.)
        if parts and parts[-1] in ['weight', 'bias']:
            parts = parts[:-1]
        
        if len(parts) >= 3:
            return '.'.join(parts[:3]) + '.'
        else:
            return '.'.join(parts) + '.'
    
    # Get unique layer names
    layer_names = set()
    for name, _ in model.named_parameters():
        layer_name = extract_layer_identifier(name)
        layer_names.add(layer_name)
    
    return len(layer_names)