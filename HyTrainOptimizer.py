import torch
import torch.nn as nn
from queue import Queue, Empty
import math 
import time
import threading
from collections import defaultdict

from ColossalAI.colossalai.kernel.kernel_loader import CPUAdamLoader

class HyTrainOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            model,
            lr=1e-3,
            bias_correction=True,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            adamw_mode=True,
        ):
        default_args = dict(lr=lr, betas=betas, eps=eps, bias_correction=bias_correction, weight_decay=weight_decay)

        # Initialize CUDA streams only if CUDA is available
        self.device_available = torch.cuda.is_available()
        if self.device_available:
            self.grad_transfer_stream = torch.cuda.Stream()
            self.param_transfer_stream = torch.cuda.Stream()
            self.optimizer_state_transfer_stream = torch.cuda.Stream()
            self.optimizer_state_update_stream = torch.cuda.Stream()
        else:
            self.grad_transfer_stream = None
            self.param_transfer_stream = None
            self.optimizer_state_transfer_stream = None
            self.optimizer_state_update_stream = None

        # Thread synchronization
        self.done = threading.Event()
        self.threads_started = threading.Event()
        
        # Enhanced k-factor tracking
        self.k_factor = 0.0
        self.k_factor_history = []  # Track k-factor changes over time
        self.last_k_update_step = 0
        self.min_steps_between_updates = 5  # Minimum steps between k-factor updates
        
        param_groups_custom = []
        total_unique_params = 0
        seen_params = set()
        param_group_counter = 0

        # Build parameter groups with proper error handling
        for name, module in model.named_modules():
            try:
                module_params = list(module.parameters(recurse=False))
                unique_params = [p for p in module_params if id(p) not in seen_params and p.requires_grad]
                if unique_params:
                    param_group_counter += 1
                    param_groups_custom.append({
                        "params": unique_params,
                        "name": name,
                        "param_group_number": param_group_counter,
                        "update_device": "cpu"  # default the parameter group to cpu initially
                                            # during training, there can be change in the device.
                    })
                    total_unique_params += sum(p.numel() for p in unique_params)
                    seen_params.update(id(p) for p in unique_params)
            except Exception as e:
                print(f"Error processing module {name}: {e}")
                continue

        self.total_param_groups = len(param_groups_custom)
        super(HyTrainOptimizer, self).__init__(param_groups_custom, default_args)
        
        # Initialize CPU Adam with error handling
        try:
            cpu_adam = CPUAdamLoader().load()
            self.cpu_adam_op = cpu_adam.CPUAdamOptimizer(lr, betas[0], betas[1], eps, weight_decay, adamw_mode)
        except Exception as e:
            print(f"Warning: Failed to load CPU Adam optimizer: {e}")
            self.cpu_adam_op = None


        self.param_dict = {}
        self.param_group_map = {}
        self.adamw_mode = adamw_mode


        self.gradient_transfer_queue_cpu = Queue()
        self.gradient_transfer_queue_gpu = Queue()
        self.param_transfer_queue = Queue()
        self.param_update_queue_cpu = Queue()
        self.param_update_queue_gpu = Queue()


        self._initialize_param_dict()
        self._initialize_layer_mapping(model)

        # Performance tracking
        self.param_update_time_with_enqueue = 0
        self.param_update_time_wo_enqueue = 0
        self.has_split = False
        
        # Enhanced performance metrics for continuous k-factor updates
        self.recent_backward_times = []
        self.recent_update_times = []
        self.performance_window_size = 10
        
        print(f"=== Total Layers in the model: {self.total_layers}")
        print(f"=== Total parameter groups: {self.total_param_groups}")

    def _initialize_param_dict(self):
        """Initialize parameter dictionary with proper memory management"""
        current_id = 1
        for group in self.param_groups:
            for p in group["params"]:
                try:
                    event = threading.Event()
                    event.set()
                    self.param_group_map[p] = group
                    
                    # Use CPU pinned memory for better transfer performance
                    
                    self.param_dict[p] = {
                        "param_id": current_id,
                        "param": p.cpu().pin_memory() if group["update_device"] == "cpu" else p,
                        "grad": torch.zeros_like(p, device="cpu").pin_memory() if group["update_device"] == "cpu" else torch.zeros_like(p),
                        "ready": event,
                        "param_group_number": group["param_group_number"],
                        "checked": False,
                        "lock": threading.Lock()  # Add thread safety
                    }
                    current_id += 1
                except Exception as e:
                    print(f"Error initializing parameter {current_id}: {e}")
                    current_id += 1
                    continue

    def _initialize_layer_mapping(self, model):
        """Initialize layer mapping with error handling"""
        
        layer_name_set = set()
        self.layer_name_id_map = {}
        self.tensor_id_name_map = {}
        self.total_tensors_per_layer = defaultdict(int)
        current_tensor_id = 1

        try:
            for name, _ in model.named_parameters():
                self.tensor_id_name_map[current_tensor_id] = name
                
                # Use the new layer identification logic
                layer_name = self.extract_layer_identifier(name)
                layer_name_set.add(layer_name)
                
                # Only assign ID if this layer hasn't been seen before
                if layer_name not in self.layer_name_id_map:
                    self.layer_name_id_map[layer_name] = len(self.layer_name_id_map) + 1

                self.total_tensors_per_layer[self.get_layer_id_from_tensor_id(current_tensor_id)] += 1
                current_tensor_id += 1

        except Exception as e:
            print(f"Error in layer mapping initialization: {e}")
            
        self.total_layers = max(1, len(layer_name_set))  # Ensure at least 1 layer
        print("total layers: ", self.total_layers)
    
    def extract_layer_identifier(self, param_name):
            """Extract layer identifier - group all tensors from same transformer block"""
            parts = param_name.split(".")
            
            # Handle Llama-style: model.layers.18.* -> model.layers.18.
            if 'layers' in parts:
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = i + 1
                        break
                if layer_idx is not None:
                    return '.'.join(parts[:layer_idx + 1]) + '.'  # Just up to the layer number
            
            # Handle GPT-style: model.transformer.h.18.* -> model.transformer.h.18.
            elif 'h' in parts:
                h_idx = None
                for i, part in enumerate(parts):
                    if part == 'h' and i + 1 < len(parts) and parts[i + 1].isdigit():
                        h_idx = i + 1
                        break
                if h_idx is not None:
                    return '.'.join(parts[:h_idx + 1]) + '.'  # Just up to the layer number
            
            # For non-transformer blocks (embeddings, final norms, etc.), use the first few parts
            # Remove weight/bias suffix for consistency
            if parts and parts[-1] in ['weight', 'bias']:
                parts = parts[:-1]
            
            # Use first 2-3 parts depending on structure
            if len(parts) >= 3:
                return '.'.join(parts[:3]) + '.'
            else:
                return '.'.join(parts) + '.'

    def get_layer_id_from_tensor_id(self, p_id):
        tensor_name = self.tensor_id_name_map[p_id]
        layer_name = self.extract_layer_identifier(tensor_name)
        return self.layer_name_id_map.get(layer_name, 0)  # Use get() to avoid KeyError
    
    def should_update_k_factor(self, current_step):
        """Determine if k-factor should be updated based on step frequency and performance stability"""
        steps_since_last_update = current_step - self.last_k_update_step
        
        # Don't update too frequently
        if steps_since_last_update < self.min_steps_between_updates:
            return False
            
        # Have enough performance data
        if len(self.recent_backward_times) < 3 or len(self.recent_update_times) < 3:
            return False
            
        return True
    
    def update_k_factor(self, backward_time, alpha, update_time, current_step=None):
        """Update k-factor and reassign parameters to GPU/CPU with enhanced logic"""
        try:
            if update_time <= 0:
                print("Warning: Invalid update_time, skipping k-factor update")
                return
            
            # Calculate new k-factor
            new_k_factor = max(0, min(1, (update_time - backward_time) / ((1 + alpha) * update_time)))
            
            # Check if significant change in k-factor warrants reassignment
            k_change_threshold = 0  # Only reassign if k-factor changes by more than 5%
            significant_change = abs(new_k_factor - self.k_factor) > k_change_threshold
            
            self.k_factor = new_k_factor
            self.k_factor_history.append(new_k_factor)
            
            if current_step is not None:
                self.last_k_update_step = current_step

            print(f"Backward: {backward_time:.4f}, Alpha: {alpha}, Update Time: {update_time:.4f}")

            total_layers_to_gpu = min(int(self.total_layers * self.k_factor) + 1, self.total_layers)
            print(f"New k factor: {self.k_factor:.4f}, GPU Layers: {total_layers_to_gpu}, Total Layers: {self.total_layers}")

            total_layers_to_gpu = 3 # Hardcoded for experiment purpose.
            print(f"total layers in GPU value = ", total_layers_to_gpu)
            # Only reassign if there's a significant change or this is the first time
            if significant_change or not hasattr(self, '_first_k_update_done'):
                if self.device_available:
                    torch.cuda.synchronize()
                
                self._clear_queues() # Clear queues before reassignment to prevent deadlocks

                with torch.cuda.stream(self.optimizer_state_transfer_stream):
                    # routine called to reassign the parameters.
                    self._reassign_parameters(total_layers_to_gpu)
                    # routine called to reassign the parameters.
                    self._reassign_parameters(total_layers_to_gpu)
                    # routine called to reassign the parameters.
                    self._reassign_parameters(total_layers_to_gpu)
                torch.cuda.synchronize()
                print("The Split has been completed!")
                self._first_k_update_done = True
            else:
                print("K-factor change not significant enough for reassignment")
            
        except Exception as e:
            print(f"Error during k-factor update: {e}")
            import traceback
            traceback.print_exc()

    def update_performance_metrics(self, backward_time, update_time):
        """Update recent performance metrics for k-factor calculation"""
        self.recent_backward_times.append(backward_time)
        self.recent_update_times.append(update_time)

        # print(self.recent_backward_times, self.recent_update_times)
        
        # Keep only recent measurements
        # if len(self.recent_backward_times) > self.performance_window_size:
        #     self.recent_backward_times.pop(0)
        # if len(self.recent_update_times) > self.performance_window_size:
        #     self.recent_update_times.pop(0)

    def get_average_performance_metrics(self):
        """Get smoothed performance metrics"""
        if not self.recent_backward_times or not self.recent_update_times:
            return 0, 0
        
        # Use median instead of mean for more robust estimation
        sorted_backward = sorted(self.recent_backward_times)
        sorted_update = sorted(self.recent_update_times)
        
        n_backward = len(sorted_backward)
        n_update = len(sorted_update)
        
        median_backward = sorted_backward[n_backward // 2] if n_backward > 0 else 0
        median_update = sorted_update[n_update // 2] if n_update > 0 else 0
        
        return median_backward, median_update

    def _clear_queues(self):
        # Clear all queues to prevent deadlocks during reassignment
        queues = [
            self.gradient_transfer_queue_cpu,
            self.gradient_transfer_queue_gpu,
            self.param_transfer_queue,
            self.param_update_queue_cpu,
            self.param_update_queue_gpu
        ]
        
        for queue in queues:
            try:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
            except Exception as e:
                print(f"Error clearing queue: {e}")

    def _reassign_parameters(self, total_layers_to_gpu):
        # Reassign parameters between CPU and GPU.

        try:
            reassignment_count = 0
            for group in self.param_groups:
                layer_id = self.get_layer_id_from_tensor_id(group["param_group_number"])
                should_be_gpu = layer_id <= total_layers_to_gpu
                
                if group["update_device"] == "cpu" and should_be_gpu:
                    group["update_device"] = "cuda" # Move from CPU to GPU
                    for p in group["params"]:
                        self._move_param_to_gpu(p)
                        reassignment_count += 1
                        print(f"Moved tensor from Layer {layer_id} To GPU!")
                        
                elif group["update_device"] == "cuda" and not should_be_gpu:
                    # Move from GPU to CPU
                    group["update_device"] = "cpu"
                    for p in group["params"]:
                        self._move_param_to_cpu(p)
                        reassignment_count += 1
                        print(f"Moved tensor from Layer {layer_id} To CPU!")
            
            print(f"Total parameters reassigned: {reassignment_count}")
        except Exception as e:
            print(f"Error during parameter reassignment: {e}")
    
    def _move_param_to_gpu(self, p):
        # Move parameter and its state to GPU with proper error handling.

        try:
            # with self.param_dict[p]["lock"]:
                # for each parameter, iterate through the optimizer states and transfer it to GPU.
            if p in self.state: 
                for key in self.state[p]:
                    if torch.is_tensor(self.state[p][key]):
                        self.state[p][key] = self.state[p][key].cuda(non_blocking=True)

            # Update param_dict
            old_param_dict = self.param_dict[p]
            event = threading.Event()
            event.set()

            self.param_dict[p].update({
                "param": p.cuda() if not p.is_cuda else p,
                "grad": torch.zeros_like(p, device="cuda", dtype=p.dtype),
                "ready": event,
                "checked": True,
            })
        except Exception as e:
            print(f"Error moving parameter to GPU: {e}")
    
    def _move_param_to_cpu(self, p):
        # move parameter and its state to CPU with proper error handling
        try:
            # with self.param_dict[p]["lock"]:
            # Move optimizer state to CPU
            if p in self.state:
                for key in self.state[p]:
                    if torch.is_tensor(self.state[p][key]):
                        self.state[p][key] = self.state[p][key].cpu().pin_memory()

            # Update param_dict
            old_param_dict = self.param_dict[p]
            event = threading.Event()
            event.set()

            self.param_dict[p].update({
                "param": p.cpu().pin_memory() if p.is_cuda else p,
                "grad": torch.zeros_like(p, device="cpu", dtype=p.dtype).pin_memory(),
                "ready": event,
                "checked": True,
            })
            
            # Clean up GPU memory
            if self.device_available:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error moving parameter to CPU: {e}")
    
    def torch_adam_update(
            self,
            data,
            grad,
            exp_avg,
            exp_avg_sq,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            use_adamw=False
        ):
        # torch adam will be called only for the GPU side parameters using its gpu side 
        # optimizer states since they do not support CPU adam.
        try:
            if grad.device != data.device:
                grad = grad.to(data.device, non_blocking=True)
                print(f"Transfer to: ", data.device)

            
            if weight_decay != 0:
                if use_adamw:
                    data.mul_(1 - lr * weight_decay)
                else:
                    grad = grad.add(data, alpha=weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            data.addcdiv_(exp_avg, denom, value=-step_size)
        except Exception as e:
            print(f"Error in torch_adam_update: {e}")
            raise
    
    def _param_step(self, param_key, grad, closure=None, div_scale=-1):
        # device agnostic update function for both CPU and GPU side updates.
        # Torch Adam can be called for BF16 parameters and GPU side updates.
        # CPU Adam will be called otherwise.

        try:
            p_id = self.param_dict[param_key]["param_id"]
            l_id = self.get_layer_id_from_tensor_id(p_id=p_id)
                
            # using the GPU tensor reference, we get the device which the parameter belongs to and 
            # accordingly call CPU Adam or Torch Adam.

            group = self.param_group_map[param_key]
            param = self.param_dict[param_key]["param"] 
            device = "cuda" if group["update_device"] == "cuda" else "cpu"
            
            state = self.state[param_key]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, device=device, dtype=param.dtype)
                state["exp_avg_sq"] = torch.zeros_like(param, device=device, dtype=param.dtype)
            
            state["step"] += 1
            beta1, beta2 = group["betas"]
            
            if device == "cpu":
                grad = self.param_dict[param_key]['grad']
                if grad.dtype == torch.bfloat16:
                    print("bfloat 16 gradients, torch update")
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    self.torch_adam_update(
                        param.data, grad.data, state["exp_avg"], state["exp_avg_sq"],
                        group["lr"], beta1, beta2, group["eps"], group["weight_decay"],
                        bias_correction1, bias_correction2, self.adamw_mode
                    )
                else:
                    
                    self.cpu_adam_op.step(
                        state["step"], group["lr"], beta1, beta2,
                        group["eps"], group["weight_decay"], group["bias_correction"],
                        param.data, grad.data, state["exp_avg"], state["exp_avg_sq"],
                        div_scale
                    )
            else:
                tensors_to_check = {
                    'param_key.data': param_key.data,
                    'grad.data': grad.data,
                    'state["exp_avg"]': state["exp_avg"],
                    'state["exp_avg_sq"]': state["exp_avg_sq"]
                }
                

                # this code was written to debug which tensor (needed for torch adam on GPU) is not available.
                non_gpu_tensors = []
                all_on_gpu = True
                for tensor_name, tensor in tensors_to_check.items():
                    if not tensor.is_cuda:
                        non_gpu_tensors.append(f"{tensor_name} is on {tensor.device}")
                        all_on_gpu = False

                if not all_on_gpu:
                    print(f"Not on GPU: {non_gpu_tensors}")
                    return

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                self.torch_adam_update(
                    param_key.data, grad.data, state["exp_avg"], state["exp_avg_sq"],
                    group["lr"], beta1, beta2, group["eps"], group["weight_decay"],
                    bias_correction1, bias_correction2, self.adamw_mode
                )
                # print(f"torch adam called successfully in gpu for l_id: {l_id} and p_id: {p_id}.")

        except Exception as e:
            print(f"grad device: {grad.device}, param device: {param_key.device}")
            print(f"Layer ID: {l_id}, PID: {p_id}, Update Device: {device}: Error in _param_step for {group.get('name', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
    
    def update_param_thread_function_in_cpu(self):
        # CPU parameter update thread function.

        print("CPU update thread started")
        while not self.done.is_set():
            try:
                # Here the gradient will be None, as the gradient is already present in CPU.
                param_key, _, enqueue_time = self.param_update_queue_cpu.get(block=True, timeout=1)
                if param_key is None:  # Poison pill
                    break

                param_update_cpu_start_time = time.time()
                # The gradient will be taken from the CPU side as the transfer has already been done.
                self._param_step(param_key=param_key, grad=_) # CPU Adam will be called.
                update_time = time.time() - param_update_cpu_start_time
                self.param_update_time_wo_enqueue += update_time
                
                # Update performance metrics for continuous k-factor calculation
                self.update_performance_metrics(0, update_time)  # 0 for backward time since this is update only

                try:
                    # After update completes, enqueue back to the param transfer queue, to update the GPU
                    # side copy of parameters with the updated parameters. FWD begins after this.
                    
                    self.param_transfer_queue.put(param_key)
                except Exception as e:
                    print(f"Failed to enqueue parameter for transfer: {e}")
                    import traceback
                    traceback.print_exc()

            except Empty:
                continue
            except Exception as e:
                print(f"Error in CPU update thread: {e}")
                continue
        print("CPU update thread finished")

    def update_parameters_in_gpu_routine(self):
        # this routine will be called in the main thread - so that gpu updates and 
        # the FWD-BWD happen in same thread - as developed by the cotrain framework.

        # it will be done with the assumption that the backward event will be completed 
        # before this routine's invocation.

        while not self.gradient_transfer_queue_gpu.empty():
            try:
                param_key, gradient, enqueue_time = self.gradient_transfer_queue_gpu.get(block=True, timeout=2)
                if param_key is None:
                    break

                param_update_gpu_start_time = time.time()
                self._param_step(param_key=param_key, grad=gradient)
                update_time = time.time() - param_update_gpu_start_time
                self.param_update_time_wo_enqueue += update_time
                
                # Update performance metrics for continuous k-factor calculation
                self.update_performance_metrics(0, update_time)  # 0 for backward time since this is update only

                p_id = self.param_dict[param_key]["param_id"]
                group = self.param_group_map[param_key] 
                device = "cuda" if group["update_device"] == "cuda" else "cpu"
                # print(f"Update in GPU, Device: {device}, P_ID: {p_id}")
                
                param_key.grad = None
                self.param_dict[param_key]['ready'].set()

            except Empty:
                print("No gradients available in gpu update queue.")
                break
            except Exception as e:
                raise RuntimeError(f"Error in GPU update routine", e)


    
    def communication_thread_function(self):
        """Communication thread function for handling transfers with proper error handling"""
        print("Communication thread started")
        while not self.done.is_set():
            try:
                # Priority to CPU gradient transfers
                if not self.gradient_transfer_queue_cpu.empty():
                    param_key, gradient, enqueue_time = self.gradient_transfer_queue_cpu.get(block=True, timeout=1)
                    
                    if param_key is None:  # Poison pill
                        break
                    with torch.cuda.stream(self.grad_transfer_stream):
                        self.param_dict[param_key]['grad'].copy_(gradient, non_blocking = False)
                        param_key.grad = None
                        self.param_update_queue_cpu.put((param_key, param_key.grad, enqueue_time))

                elif not self.param_transfer_queue.empty():
                    # CPU to GPU parameter transfer (lower priority)
                    
                    # access the GPU reference of the tensor from the param_transfer_queue to update
                    # its content.
                    param_key = self.param_transfer_queue.get(block=True, timeout=1) 

                    with torch.cuda.stream(self.param_transfer_stream):
                        try:
                            param_key.data = self.param_dict[param_key]["param"].to(param_key.device, non_blocking = False)
                        except Exception as e:
                            print(f"Error copying parameter data: {e}")
                            
                    # Each tensor waits until the event has been completed, FWD can not begin until this is completed.
                    self.param_dict[param_key]['ready'].set()

            except Empty:
                continue
            except Exception as e:
                print(f"Error in communication thread: {e}")
                continue
        print("Communication thread finished")
    
    def wait_for_param(self, param):
        try:
            if param not in self.param_dict:
                return
            # with self.param_dict[param]["lock"]:
            p_id = self.param_dict[param]['param_id']
            # print(f"Layer ID: {self.get_layer_id_from_tensor_id(p_id)}, Param ID: {p_id}: checked: {self.param_dict[param]['checked']}, set_or_not: {self.param_dict[param]['ready'].is_set()}")
            if not self.param_dict[param]["checked"]:

                # Wait for parameter to be ready
                ready = self.param_dict[param]["ready"].wait(timeout=10.0)  # Increased timeout
                if not ready:
                    print(f"Warning: Timeout waiting for parameter {self.param_dict[param]['param_id']} ready event")
                    self.param_dict[param]["ready"].set()  # Force set to avoid deadlock
                    
            self.param_dict[param]["ready"].clear()
            self.param_dict[param]["checked"] = True
                
        except Exception as e:
            print(f"Error in wait_for_param: {e}")
            # Set as checked to avoid blocking training
            if param in self.param_dict:
                self.param_dict[param]["checked"] = True

    def cleanup(self):
        # Clean up resources and stop threads gracefully

        print("Starting optimizer cleanup...")
        self.done.set()
        # Synchronize CUDA if available
        if self.device_available:
            try:
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Error during CUDA synchronization: {e}")

        print("Optimizer cleanup completed")
