import os
import torch 
import time 
import argparse
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config

from train_engine import (
    TrainerHyTrain
)

from utils import model_builder, MODEL_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for HyTrainDNN')
    
    # Common arguments
    parser.add_argument('--framework', type=str, required=True, choices=["hytrain"],
                        help='Framework to use: hytrain')
    parser.add_argument('--lr', '--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--context_length', type=int, default=1024,
                        help='Context length for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs to train (default: 8)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps to train per epoch (default: 10)')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Number of warmup steps to train per epoch (default: 10)')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training with bf16.')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to be able to load larger models on GPU VRAM.')
    parser.add_argument('--model', type=str, default="llama",
                        help='Enter the model configuration. Ex: llama, gpt2')
    parser.add_argument('--model-size', type=str, default="7b",
                        help='Enter the model configuration. Ex: 100m, 3b, 5b, 7b')
    parser.add_argument('--dataset', type=str, default="c4",
                        help='Enter the dataset you wish to use. Options: c4, wikistack (Wikipedia + Stack Exchange).')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Enter the path of the model checkpoint folder.')
    parser.add_argument('--opt_checkpoint_path', type=str, default=None,
                        help='Enter the path of the optimizer checkpoint folder.')
    parser.add_argument('--metadata_checkpoint_path', type=str, default=None,
                    help='Enter the path of the metadata checkpoint file.')
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Enter the path of the folder with all models.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Set a seed for pytorch and data shuffling (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha value that has been calculated emperically for cotrain.')
    parser.add_argument('--checkpoint_interval', type=int, default=2000,
                        help='Interval in which checkpointing is needed. Ex: 2000')
    parser.add_argument('--shuffle', action='store_true',
                        help='Enable shuffling of training data')
    parser.add_argument('--shuffle_buffer_size', type=int, default=10000,
                        help='Buffer size for streaming dataset shuffling (default: 10000)')
    
    return parser.parse_args()

def set_deterministic_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def validate_args(args):
    """Validate framework-specific arguments"""
    return None

# If you wish to load weights from a path, uncomment the load_state_dict
# sub section of this code.


def load_model(args, state_dict_path):
    """Load model with consistent weights across frameworks"""
    # Set seed before model creation to ensure identical initialization
    set_deterministic_seed(args.seed)

    if args.model == "llama":
        llama_config = MODEL_CONFIGS[args.model_size]
        model = AutoModelForCausalLM.from_config(
            llama_config,
            torch_dtype=torch.float32,
        )
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
    elif args.model == "gpt2":
        model = model_builder(args.model_size)(context_length = args.context_length)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {round(total_params/1e9, 2)}B")
    return model

def create_trainer(args, model):
    """Create trainer based on framework with consistent parameters"""
    base_params = {
        'model': model,
        'model_str': f"{args.model}_{args.model_size}_{args.batch_size}_{args.context_length}",
        'lr': args.lr,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_steps': args.steps,
        'warmup_steps': args.warmup_steps,
        'use_amp': args.use_mixed_precision,
        'dataset': args.dataset,
        'context_length': args.context_length,
        'model_checkpoint_path': args.model_checkpoint_path,
        'opt_checkpoint_path': args.opt_checkpoint_path,
        'checkpoint_interval': args.checkpoint_interval,
        # 'seed': args.seed,  # Pass seed to trainer for data consistency
        # 'shuffle': args.shuffle,
        # 'shuffle_buffer_size': args.shuffle_buffer_size,
    }


    if args.framework == "hytrain":
        hytrain_params = {
            "alpha": args.alpha
        }
        base_params.update(hytrain_params)
        return TrainerHyTrain(
            **base_params
        )

    else: 
        raise NotImplementedError

def main():
    args = parse_args()
    
    print(f"===== Running {args.framework.upper()} Framework =====")
    for arg in vars(args):
        print(f"=====  {arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    # Validate arguments and paths
    state_dict_path = validate_args(args)
    
    # Set deterministic seed for reproducibility
    set_deterministic_seed(args.seed)
    print(f"✅ Set deterministic seed: {args.seed}")
    
    # Load model with consistent weights
    model = load_model(args, state_dict_path)
    print(model)
    
    # Create trainer based on framework
    trainer = create_trainer(args, model)
    print(f"✅ Created {args.framework.upper()} trainer with consistent configuration")
    
    # Start training
    print(f"🚀 Starting {args.framework.upper()} training...")
    trainer.train()

if __name__ == "__main__":
    main()
