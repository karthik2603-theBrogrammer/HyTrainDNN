import os
import csv
import time
import hashlib
import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import warnings
import resource
import math
import threading
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from transformers import AutoTokenizer

from HyTrainOptimizer import HyTrainOptimizer

from utils import SkipParObjectiveMinimizer, SkipParEpochManager, SkipparDataLoader, GPTLMLoss, get_model_numel, format_numel_str, get_layers_in_model


class TrainerHyTrain(object):
    def __init__(
        self, 
        model, 
        model_str,
        lr, 
        num_epochs, 
        batch_size,
        num_steps, 
        warmup_steps, 
        use_amp, 
        dataset,
        context_length,
        alpha,
        k_update_frequency=20, 
        model_checkpoint_path = None,
        opt_checkpoint_path = None,
        checkpoint_interval = 2000,

    ):
        self.device = "cuda"
        self.lr = lr
        self.model_str = model_str
        self.use_amp = use_amp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.resumed_from_checkpoint = model_checkpoint_path is not None and opt_checkpoint_path is not None
        self.num_model_params = round(sum(p.numel() for p in model.parameters())/1e9, 2)

        # Time tracking attributes (similar to SkipPar)
        self.elapsed_time_at_checkpoint = 0
        self.loss_time_history = []
        self.time_spent_in_checkpointing = 0

        self.gpt_loss_fn = GPTLMLoss()
        self.criterion = nn.CrossEntropyLoss()
        
        # Model and Optimizer initialization with checkpointing
        if model_checkpoint_path is not None:
            model = self.load_model_checkpoint(model, model_checkpoint_path)
            
        model = model.to("cuda")
        self.model = model
        
        # Optimizer initialization
        if opt_checkpoint_path is not None:
            self.optimizer = self.load_optimizer_checkpoint(model, lr=lr, optimizer_checkpoint_path=opt_checkpoint_path)
        else:
            self.optimizer = HyTrainOptimizer(model, lr=lr)
            
        print("optimizer initialized")
        
        self.param_dict = self.optimizer.param_dict
        self.param_group_map = self.optimizer.param_group_map

        self.fwd_bwd_thread = None
        self.update_thread_cpu = None
        self.communication_thread = None
        
        # Create threads
        self._initialize_threads()

        # Attach the Hooks
        self._attach_grad_transfer_hook()
        self._attach_wait_for_param_hook()

        self.epochs = num_epochs
        self.steps = num_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.batch_size = batch_size
        self.context_length = context_length
        self.dataset = dataset
        self.checkpoint_interval = checkpoint_interval
        self.k_update_frequency = k_update_frequency

        # Times - Enhanced for continuous k-factor calculation
        self.forward_step_time = 0
        self.backward_step_time = 0

        self.current_step = 0
        self.global_step = 0
        self.current_epoch = 0
        self.has_split = False

        # Gradient Scaling Attributes
        self.cotrain_grad_scaler = torch.amp.GradScaler("cuda",
            init_scale = 2.0**16,
            growth_factor = 2.0,
            backoff_factor = 0.5,
            growth_interval = 2000,
            enabled = True
        )

        self.found_inf = torch.full((), 0.0, dtype=torch.float32, device="cuda")
        self.cotrain_grad_scaler._scale = torch.full((), self.cotrain_grad_scaler._init_scale, dtype=torch.float32, device="cuda")
        self.cotrain_grad_scaler._growth_tracker = torch.full((), self.cotrain_grad_scaler._init_growth_tracker, dtype=torch.int32, device="cuda")
        self.inv_scale = self.cotrain_grad_scaler._scale.double().reciprocal().float()

        self.total_found_infs_per_it = 0

        # Start time reference for elapsed time calculation
        self.start_time_reference = time.time()

        print("=== Timestamp: ", self.timestamp)

    def load_model_checkpoint(self, model, model_checkpoint_path):
        """Load model checkpoint similar to SkipPar implementation"""
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        self.global_step = checkpoint.get('global_step')
        self.current_epoch = checkpoint.get('current_epoch')
        self.current_step = checkpoint.get('current_step')
        self.alpha = checkpoint.get('alpha', self.alpha)
        self.has_split = checkpoint.get('has_split', False)
        
        self.elapsed_time_at_checkpoint = checkpoint.get('elapsed_time_at_checkpoint')
        self.time_spent_in_checkpointing = checkpoint.get('time_spent_in_checkpointing')
        self.loss_time_history = checkpoint.get('loss_time_history', [])
        
        self.timestamp = checkpoint['timestamp']
        
        # Restore gradscaler attributes
        if 'grad_scaler_scale' in checkpoint:
            self.cotrain_grad_scaler._scale.fill_(checkpoint['grad_scaler_scale'])
        if 'grad_scaler_growth_tracker' in checkpoint:
            self.cotrain_grad_scaler._growth_tracker.fill_(checkpoint['grad_scaler_growth_tracker'])
            
        print(f"=== Checkpoint loaded. Resuming from epoch {self.current_epoch}, step {self.current_step}, global_step {self.global_step}.")
        print(f"INFO: Checkpoint === ")
        for k, v in checkpoint.items():
            if k != "model_state_dict":
                print(f"=== {k}: {v}.")
                
        del checkpoint
        return model

    def load_optimizer_checkpoint(self, model, lr, optimizer_checkpoint_path):
        """Load optimizer checkpoint similar to SkipPar implementation"""
        optimizer = HyTrainOptimizer(model, lr=lr)
        opt_checkpoint = torch.load(optimizer_checkpoint_path)
        
        # Restore optimizer state
        for param_key in model.parameters():
            param = optimizer.param_dict[param_key]["param"]
            param_id = optimizer.param_dict[param_key]["param_id"]
            state = optimizer.state[param]

            opt_param_state = opt_checkpoint[param_id]
            state["step"] = opt_param_state["step"]
            state["exp_avg"] = opt_param_state["exp_avg"]
            state["exp_avg_sq"] = opt_param_state["exp_avg_sq"]
        
        # Restore CoTrain-specific optimizer state if available
        if 'k_factor' in opt_checkpoint:
            optimizer.k_factor = opt_checkpoint['k_factor']
        if 'param_group_map' in opt_checkpoint:
            optimizer.param_group_map = opt_checkpoint['param_group_map']
            
        del opt_checkpoint
        return optimizer

    def save_checkpoint(self, loss, checkpoint_dir):
        """Save checkpoint with comprehensive state similar to SkipPar"""
        checkpoint_start_time = time.time()
        
        loss = round(loss, 2)
        checkpoint_dir += f"/{self.model_str}/model={str(self.model.model)[:10]}_step={self.global_step}_loss={loss}_{self.timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'loss': loss,
            'alpha': self.alpha,
            'has_split': self.has_split,
            
            'elapsed_time_at_checkpoint': self.get_elapsed_time(),
            'time_spent_in_checkpointing': self.time_spent_in_checkpointing,
            'loss_time_history': self.loss_time_history,
            
            'grad_scaler_scale': self.cotrain_grad_scaler._scale.item(),
            'grad_scaler_growth_tracker': self.cotrain_grad_scaler._growth_tracker.item(),
            
            'timestamp': self.timestamp,
        }
        
        for k, v in checkpoint.items():
            if k != 'model_state_dict':
                print(f"{k}: {v}")
                
        mod_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mod_step_{self.global_step}.pt')
        torch.save(checkpoint, mod_checkpoint_path)
        
        # Save optimizer state
        opt_ckp_dict = {}
        for param_key in self.model.parameters():
            param = self.optimizer.param_dict[param_key]["param"]
            param_id = self.optimizer.param_dict[param_key]["param_id"]
            state = self.optimizer.state[param]

            if param_id not in opt_ckp_dict:
                opt_ckp_dict[param_id] = {}

            opt_ckp_dict[param_id]["step"] = state["step"]
            opt_ckp_dict[param_id]["exp_avg"] = state["exp_avg"]
            opt_ckp_dict[param_id]["exp_avg_sq"] = state["exp_avg_sq"]
        
        # Save CoTrain-specific optimizer state
        opt_ckp_dict['k_factor'] = getattr(self.optimizer, 'k_factor', None)
        opt_ckp_dict['param_group_map'] = self.optimizer.param_group_map
            
        opt_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_opt_step_{self.global_step}.pt')
        torch.save(opt_ckp_dict, opt_checkpoint_path)
        
        checkpoint_end_time = time.time()
        self.time_spent_in_checkpointing += (checkpoint_end_time - checkpoint_start_time)
        
        print(f"\n === Checkpoint saved for step: {self.global_step}.")

    def get_elapsed_time(self):
        """Calculate elapsed time excluding checkpointing time"""
        return self.elapsed_time_at_checkpoint + (time.time() - self.start_time_reference) - self.time_spent_in_checkpointing
        
    def _initialize_threads(self):
        """Initialize all threads"""
        self.fwd_bwd_thread = threading.Thread(
            target=self.fwd_bwd_thread_function, 
            daemon=True, 
            name="ForwardBackward"
        )
        self.update_thread_cpu = threading.Thread(
            target=self.update_thread_cpu_function, 
            daemon=True,
            name="CPUUpdate"
        )
        self.communication_thread = threading.Thread(
            target=self.communication_thread_function, 
            daemon=True,
            name="Communication"
        )

    def _attach_grad_transfer_hook(self):
        def grad_transfer_hook(param):
            try:
                if param.grad is not None:
                    with torch.no_grad():
                        torch._amp_foreach_non_finite_check_and_unscale_(
                                [param.grad],
                                self.found_inf,
                                self.inv_scale,
                            )
                        if(not self.found_inf):
                            pass
                        else:
                            self.optimizer.param_dict[param]["ready"].set()
                            return None
                        
                    update_device = self.param_group_map[param]['update_device']

                    if update_device == "cuda":
                        self.optimizer.gradient_transfer_queue_gpu.put(
                            (param, param.grad, time.time())
                        )
                    elif update_device == "cpu":
                        self.optimizer.gradient_transfer_queue_cpu.put(
                            (param, param.grad, time.time())
                        )
                else:
                    self.optimizer.param_dict[param]["ready"].set()
                return None 
            except Exception as e:
                print(f"Error in grad_transfer_hook: {e}")
            return None
    
        # register the tensor level backward hook to all the tensors of the llm.
        for _, param in self.model.named_parameters():
                param.register_post_accumulate_grad_hook(grad_transfer_hook)

    def _attach_wait_for_param_hook(self):
        def wait_param_hook(module, _):
            try:
                for _, param in module.named_parameters():
                    self.optimizer.wait_for_param(param)

            except Exception as e:
                print(f"Error in wait_param_hook: {e}")
            return None
        
        for module in self.model.modules():
            if(next(module.children(), None) is None):
                module.register_forward_pre_hook(wait_param_hook)
        return None

    def fwd_bwd_thread_function(self):
        try:
            # load the tokenizer.
            if "gpt2" in  self.model_str:
                tokenizer = AutoTokenizer.from_pretrained(os.getenv("GPT2_TOKENIZER"))
                print("Loaded GPT2 Tokenizer.")
            elif "llama" in self.model_str:
                tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLAMA_TOKENIZER"))
                print("Loaded Llama Tokenizer.")
            else:
                raise NotImplementedError(f"Tokenizer for the model {self.model_str} is not supported.")

            # load the data loader.
            if self.dataset == "wikistack":
                print(f"Using wikistack dataset.")
                dataloader = SkipparDataLoader(
                tokenizer=tokenizer,
                context_size=self.context_length,
                batch_size=self.batch_size
            ).build_wikistack_dataloader(
                shuffle=True,
                skip_steps=self.global_step
            )
            elif self.dataset == "c4":
                print(f"Using c4 dataset.")
                dataloader = SkipparDataLoader(
                    tokenizer=tokenizer,
                    context_size=self.context_length,
                    batch_size=self.batch_size    
                ).build_c4_dataloader(shuffle=True, skip_steps=self.global_step)
            elif self.dataset == "slim_pajama":
                print("Using slim pajama dataset")
                dataloader = SkipparDataLoader(
                    tokenizer= tokenizer,
                    context_size= self.context_length,
                    batch_size= self.batch_size
                ).build_slim_pajama_dataloader(shuffle= True, skip_steps= self.global_step)
            else:
                raise NotImplementedError(f"{self.dataset} not supported yet.")

            ##############################     WARM UP     ##############################    
            dl = iter(dataloader)
            dl._dataset.set_epoch(42)

            if not self.resumed_from_checkpoint:
                # Run warmup only if not resuming from checkpoint
                for step in tqdm(range(self.warmup_steps)):

                    with torch.autocast(device_type = self.device, dtype = torch.bfloat16, enabled = self.use_amp):
                        batch = next(dl)
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        labels = batch["labels"]

                        input_ids = input_ids.to("cuda")
                        attention_mask = attention_mask.to("cuda")
                        labels = labels.to("cuda")

                        if "gpt2" in self.model_str:
                            outputs = self.model(input_ids, attention_mask)
                            loss = self.gpt_loss_fn(outputs, labels)
                        elif "llama" in self.model_str:
                            outputs = self.model(input_ids, attention_mask, labels=labels)
                            loss = outputs.loss
                        else:
                            raise RuntimeError("model not supported for training")

                    del outputs # clear activation memory.

                    torch.cuda.current_stream().synchronize()

                    backward_start = time.time()
                    self.cotrain_grad_scaler.scale(loss).backward()

                    # call the gpu parameter's update routine in the main thread, as implemented in cotrain.
                    update_start = time.time()
                    self.optimizer.update_parameters_in_gpu_routine() 
                    update_time = time.time() - update_start
                    
                    backward_time = time.time() - backward_start
                    self.backward_step_time += backward_time
                    
                    print(f"warmup step: {step + 1}, loss: {loss}", end = "")

                    torch.cuda.current_stream().synchronize()

                    torch._amp_update_scale_(
                        self.cotrain_grad_scaler._scale,
                        self.cotrain_grad_scaler._growth_tracker,
                        torch.tensor([self.total_found_infs_per_it], device = "cuda", dtype = torch.float32),
                        self.cotrain_grad_scaler._growth_factor,
                        self.cotrain_grad_scaler._backoff_factor,
                        self.cotrain_grad_scaler._growth_interval,
                    )

                # Initial k-factor calculation after warmup
                if self.warmup_steps > 0:
                    tB = self.backward_step_time / self.warmup_steps
                    tU_wo_q = self.optimizer.param_update_time_wo_enqueue / self.warmup_steps
                    alpha = self.alpha
                    
                    self.optimizer.update_k_factor(
                        backward_time=tB, 
                        alpha=alpha, 
                        update_time=tU_wo_q
                    )
            ########################################### WARM UP END ################################################    

            # Main training loop with proper checkpoint resumption
            training_start_epoch = self.current_epoch
            training_end_epoch = self.epochs + 1 if self.resumed_from_checkpoint else self.epochs
            first_epoch_since_checkpoint_resumed = self.resumed_from_checkpoint

            for epoch in range(training_start_epoch, training_end_epoch):
                if self.resumed_from_checkpoint and first_epoch_since_checkpoint_resumed:
                    first_epoch_since_checkpoint_resumed = False
                else:
                    self.current_epoch += 1
                    
                dl = iter(dataloader)
                dl._dataset.set_epoch(self.current_epoch)

                start_step = self.current_step if self.resumed_from_checkpoint else 0
                tqdm_object = tqdm(range(start_step, self.steps + 1), 
                       initial=start_step, 
                       total=self.steps + 1,
                       desc=f"Epoch {self.current_epoch}")

                for step in tqdm_object:
                    try:    
                        self.current_step = step + 1
                        self.global_step += 1

                        with torch.autocast(device_type = self.device, dtype = torch.bfloat16, enabled = self.use_amp):
                            batch = next(dl)
                            input_ids = batch["input_ids"]
                            attention_mask = batch["attention_mask"]
                            labels = batch["labels"]

                            input_ids = input_ids.to("cuda")
                            attention_mask = attention_mask.to("cuda")
                            labels = labels.to("cuda")

                            if "gpt2" in self.model_str:
                                outputs = self.model(input_ids, attention_mask)
                                loss = self.gpt_loss_fn(outputs, labels)
                            elif "llama" in self.model_str:
                                outputs = self.model(input_ids, attention_mask, labels=labels)
                                loss = outputs.loss
                            else:
                                raise RuntimeError("model not supported for training")
                        

                        del outputs # clear activation memory.


                        torch.cuda.current_stream().synchronize()

                        # reset this parameters for every step.
                        self.backward_step_time = 0
                        self.optimizer.param_update_time_wo_enqueue = 0

                        backward_start = time.time()
                        self.cotrain_grad_scaler.scale(loss).backward()
                        backward_time = time.time() - backward_start
                        self.backward_step_time  = backward_time

                        # GPU: Call the gpu parameter's update routine in the main thread.
                        self.optimizer.update_parameters_in_gpu_routine()

                        # Update k-factor every N steps
                        if self.current_step % self.k_update_frequency == 0:
                            backward_time_for_k = self.backward_step_time
                            update_time_for_k = self.optimizer.param_update_time_wo_enqueue
                            if update_time_for_k > 0:  # Avoid division by zero
                                self.optimizer.update_k_factor(
                                    backward_time=backward_time_for_k,
                                    alpha=self.alpha,
                                    update_time=update_time_for_k
                                )
                                print(f" | K-factor updated at step {self.current_step}")

                        # Store loss and elapsed time for tracking
                        elapsed_time = self.get_elapsed_time()
                        self.loss_time_history.append((self.global_step, loss.item(), elapsed_time))
                        
                        if self.global_step % 200 == 0:
                            self.save_data_to_csv(self.global_step)

                        if self.global_step % self.checkpoint_interval == 0:
                            self.save_checkpoint(loss=loss.item(), checkpoint_dir="ckp")
                        
                        torch.cuda.current_stream().synchronize()
                        
                        torch._amp_update_scale_(
                            self.cotrain_grad_scaler._scale,
                            self.cotrain_grad_scaler._growth_tracker,
                            torch.tensor([self.total_found_infs_per_it], device = "cuda", dtype = torch.float32),
                            self.cotrain_grad_scaler._growth_factor,
                            self.cotrain_grad_scaler._backoff_factor,
                            self.cotrain_grad_scaler._growth_interval,
                        )
                
                    except StopIteration:
                        print(f"DataLoader exhausted at step {step}.")
                        break
                    except Exception as e:
                        print(f"Error in training step {step}: {e}.")
                        continue
            
                print(f"Epoch {self.current_epoch} completed.")
                self.save_data_to_csv(self.global_step)
                # Reset step counter for next epoch
                self.current_step = 0
        
        except Exception as e:
            print(f"Error in forward-backward thread: {e}")
        finally:
            print("Forward-backward thread completed")
    
    def update_thread_cpu_function(self):
        try:
            self.optimizer.update_param_thread_function_in_cpu()
        except Exception as e:
            print(f"Error in CPU update thread: {e}")

    def communication_thread_function(self):
        try:
            self.optimizer.communication_thread_function()
        except Exception as e:
            print(f"Error in communication thread: {e}")

    def train(self):
        try:
            print("Starting training threads...")
            
            # Start all threads
            self.fwd_bwd_thread.start()
            self.update_thread_cpu.start()
            self.communication_thread.start()

            # Wait for main training thread to complete
            self.fwd_bwd_thread.join()
            
            # Signal other threads to stop
            self.optimizer.done.set()
            
            threads = [self.update_thread_cpu, self.communication_thread]
            for thread in threads:
                thread.join(timeout=10.0)
                if thread.is_alive():
                    print(f"Warning: Thread {thread.name} did not terminate gracefully")
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            self.optimizer.done.set()  # Signal threads to stop
            raise
    
    def save_data_to_csv(self, step_number):
        """Save loss data with step numbers and elapsed time similar to SkipPar"""
        output_folder = f"CoTrain_{str(self.model.model)[:10]}-BS{self.batch_size}-{self.num_model_params}B_{self.timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        loss_dir = os.path.join(output_folder, "loss")
        os.makedirs(loss_dir, exist_ok=True)
        loss_csv_path = os.path.join(loss_dir, "loss_curve.csv")

        file_exists = os.path.exists(loss_csv_path)
        is_resuming = self.resumed_from_checkpoint
        
        with open(loss_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists and not is_resuming:
                writer.writerow(['Step', 'Loss', 'Elapsed_Time'])

            # Write loss values with their corresponding step numbers and elapsed time
            for step, loss_val, elapsed_time in self.loss_time_history:
                writer.writerow([step, loss_val, elapsed_time])
                
        print(f"CSV data saved at step {step_number}. Added {len(self.loss_time_history)} loss values.")
        self.loss_time_history.clear()
