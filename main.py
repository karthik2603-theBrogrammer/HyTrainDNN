

# def main(config: Dict):
    # Main function that performs the training.

    # unique_id = some_id
    # if some_id name already in model_checkpoints,
    # load from that directory
    # else, prepare checkpoints
    # prepare_checkpoints(config=dnn_schema)

    # trainer = Trainer(config= config)
    # trainer.train()

    # start the layer by layer training.
    # 1. Fetch from disk
    # 2. Send from CPU to GPU
    # 3. Perform forward pass on the GPU.
    # 4. Store a copy of that activation to the CPU (checkpointing to the CPU)
    # ....
    # 5. Forward pass completed; we have the loss
    # 6. Bring Layer(n) from Disk to CPU to GPU again, 
    # 7. Start backward pass from Layer(n)
    # 8. Find gradients using the activations, loss (TBD)
    # 9. Update the parameters using W(new) = W(old) - LR * gradients. 
    # 10. Store the layer back to disk.
    # for layer_config in config:
    #     pass
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from tqdm import tqdm


from utils import prepare_checkpoints, LossFn, create_layer

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


class Trainer:
    def __init__(self, config: List[Dict], checkpoint_dir: str, learning_rate: float = 2e-5, device: str = "cuda"):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Storage for activations and gradients (CPU-side checkpointing)
        self.layer_activations = []
        self.layer_gradients = []
        
        # Fixed training data
        self.batch_size = 1024
        self.seq_length = 2048

        torch.manual_seed(42)

        # passing input of size 1024 * 512
        # output of dnn is 1024 * 10 => logits, cross entropy loss used.
        self.input_ids = torch.randint(0, 100, (dnn_schema[0]["input"], dnn_schema[0]["input"]), dtype = torch.float32)
        self.labels = torch.randint(0, 1, (dnn_schema[0]["input"], ), dtype = torch.float32)

        print(f"Using device: {self.device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Batch size: {self.batch_size}")
        print(f"Sequence length: {self.seq_length}")
        print(f"Input shape: {self.input_ids.shape}")
        print(f"Labels shape: {self.labels.shape}")
        # print(f"First layer type: {first_layer['layer_type']}")
        # print(f"Final layer output size: {output_size}")
        
    def fetch_layer(self, layer_name: str) -> nn.Module:
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{layer_name}.pt")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        
        layer = create_layer(checkpoint['config'])
        layer.load_state_dict(checkpoint['state_dict'])
        layer = layer.to(self.device)
        layer.train()
        
        return layer
        
    def save_layer(self, layer: nn.Module, layer_name: str):
        
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{layer_name}.pt")
        
        # Move to CPU before saving
        layer_cpu = layer.cpu()
        checkpoint = {}
        checkpoint['state_dict'] = layer_cpu.state_dict()
        torch.save(checkpoint, checkpoint_file) # this stores the checkpoint with the updated weights.



    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> Dict:

        self.layer_activations = []
        x = inputs.to(self.device)
        
        self.layer_activations.append(x.detach().cpu()) # Store input activation in the CPU side
        for _, layer_config in enumerate(self.config):
            layer = self.fetch_layer(layer_config["layer_name"]) # fetch from disk.
            x = layer(x)
            self.layer_activations.append(x.detach().cpu()) # move activations to the CPU.
            del layer # clear current layer from the GPU.
            torch.cuda.empty_cache() # clear 
        logits = x
        print("Logits: ", logits.shape, "Outputs: ", self.labels.shape)
        loss = LossFn(logits, labels.to(self.device)) # using cross entropy loss.

        return {
            "loss": loss, 
            "logits": logits
        }
    
    def backward_and_update(self, loss: torch.Tensor):
        grad_output = None
        for i, layer_config in enumerate(reversed(self.config)):
            layer_idx = len(self.config) - 1 - i
            layer_number = layer_config["layer_number"]
            layer_name = layer_config["layer_name"]

            layer = self.fetch_layer(layer_name= layer_name)
            input_activation = self.layer_activations[layer_number - 1]
            output_activation = self.layer_activations[layer_number]

            # find the gradients with respect to the input and outputs, 
            # find gradients and perform in place step update.

            input_activation = input_activation.clone().to("cuda")
            output_activation = output_activation.clone().to("cuda")


            # TODO: implement the logic for finding gradients through backprpopgation
            param_gradients = None
            
            # with torch.no_grad():
            #     for param, grad in zip(layer.parameters(), param_gradients):
            #         if grad is not None and param.requires_grad:
            #             # Apply SGD update
            #             print(param.data)
            #             param.data -= self.learning_rate * grad
            #             print(param.data)
            #             updated_params += 1
            self.save_layer(layer, layer_name)
            
            # Clean up GPU memory
            del layer, input_activation, output_activation
            torch.cuda.empty_cache()
        
        print("Backward pass completed!")        
        return

       
    def train_step(self):

        print("Starting forward pass...")
        # Forward pass with fixed data
        outputs = self.forward(self.input_ids, self.labels)
        loss = outputs["loss"]
        # print(f"Loss: {loss.item():.6f}")
        print("Starting backward pass and parameter updates...")
        # self.backward_and_update(loss)
        
        return loss.item()


def main(config: List[Dict] = None):

    if config is None:
        config = dnn_schema

    checkpoint_path = "model_checkpoints"

    print("Preparing new checkpoints...")
    checkpoint_dir = prepare_checkpoints(config, checkpoint_path)

    # Initialize trainer
    trainer = Trainer(config=config, checkpoint_dir=checkpoint_dir)

    # Training configuration
    num_epochs = 10
    steps_per_epoch = 10
    epoch_losses = []
    all_losses = []

    print(f"\n{'='*60}")
    print(f"Starting Training: {num_epochs} epochs × {steps_per_epoch} steps")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        
        # Progress bar for steps within epoch
        step_pbar = tqdm(
            range(steps_per_epoch), 
            desc=f"Epoch {epoch+1:3d}/{num_epochs}",
            leave=False,
            ncols=100
        )
        
        for step in step_pbar:
            loss = trainer.train_step()
            epoch_loss_sum += loss
            all_losses.append(loss)
            
            step_pbar.set_postfix({
                'Loss': f'{loss:.6f}',
                'Avg': f'{epoch_loss_sum/(step+1):.6f}'
            })
        
        epoch_loss = epoch_loss_sum / steps_per_epoch
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Avg Loss = {epoch_loss:.6f}")

    print(f"\n{'='*60}")
    print("Training Completed!")

    print(epoch_losses)


    return trainer, epoch_losses, all_losses

if __name__ == "__main__":

    print("Starting Hybrid Training with Fixed Parameters:")
    print("   • Batch Size: 1024")
    print("   • Sequence Length: 2048") 
    print("   • Epochs: 10")
    print("   • Steps per Epoch: 10")

    trainer, epoch_losses, all_losses = main(config=dnn_schema)
