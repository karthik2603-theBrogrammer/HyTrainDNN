#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=hytrain
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=node3
#SBATCH --output=outputs/sim_%J.out

export HF_HOME=/scratch/karthick/skipar/hf_hub_cache
export OMP_NUM_THREADS=16

export HOME=/scratch/karthick/skipar/temp_home
export XDG_CACHE_HOME=/scratch/karthick/skipar/cache
export TORCH_CACHE=/scratch/karthick/skipar/torch_cache
export TORCH_EXTENSIONS_DIR=/scratch/karthick/skipar/torch_extensions
export COLOSSALAI_CACHE_DIR=/scratch/karthick/skipar/colossalai_cache


export WIKISTACK_URL="/scratch/karthick/skipar/skippar_train_1.jsonl"
export C4_URL="/scratch/kevinmahesh/mtech-project-files/datasets/c4/en"
export GPT2_TOKENIZER="/scratch/karthick/skipar/gpt2-autotok/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
export LLAMA_TOKENIZER="/scratch/karthick/skipar/llama-token/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"


# Create all necessary directories
mkdir -p $HOME
mkdir -p $XDG_CACHE_HOME
mkdir -p $TORCH_CACHE
mkdir -p $TORCH_EXTENSIONS_DIR
mkdir -p $COLOSSALAI_CACHE_DIR
mkdir -p $XDG_CACHE_HOME/colossalai/torch_extensions

module load conda 
module load cuda/11.8
module load gcc/9

cd /scratch/karthick/skipar/Comprehensive-and-Integrated-Framework-for-ML-DL-Applications/Heterogeneous-Training/HyTrainDNN
source /scratch/dheemanth/coloenv1/bin/activate

echo "===== Running HyTrainDNN Framework ====="
# -m viztracer --tracer_entries=7000000 --max_stack_depth=15 --log_torch python -u
python -u train.py \
    --framework hytrain \
    --lr 2e-5 \
    --batch_size 8 \
    --context_length 2048 \
    --epochs 3 \
    --steps 1000 \
    --warmup_steps 100 \
    --use_mixed_precision \
    --use_gradient_checkpointing \
    --model gpt2 \
    --model-size gpt2_9b \
    --dataset c4 \
    --checkpoint_interval 4000 \
    --alpha 0.083 \
    --shuffle \
    --shuffle_buffer_size 10000 \
    --seed 42 \
    --models-dir /scratch/karthick/skipar/Comprehensive-and-Integrated-Framework-for-ML-DL-Applications/Heterogeneous-Training/models
    # --model-checkpoint-path /scratch/karthick/skipar/ckp/model=LlamaModel_step=40_loss=8.01/checkpoint_step_40.pt \
    # --opt-checkpoint-path /scratch/karthick/skipar/ckp/model=LlamaModel_step=40_loss=8.01/checkpoint_opt_step_40.pt

echo "HyTrainDNN Training completed!"