# HyTrainDNN

HyTrainDNN is a heterogeneous CPU-GPU based training engine capable of training large scale Deep Neural Networks. Leverages both CPU and GPU memory parallelly. 

Training of neural networks especially large scale enterprise Level LLMs are resource intensive. For instance, training a 5B Llama2 Model at full precison (32Bits per parameter) with ADAM optimizer would use the following split for GPU memory.

Model Weights: 5B * 4 Bytes per parameter = 20GB VRAM
Optimizer States: 5B * 2 (Momentum, Variance) * 4 Bytes per parameter = 40GB VRAM
Not to mention the memory taken by activations (non model data) and gradients generated during backpropogation.

Due to these factors, training large scale models remains a challenge.

Owing to this, I introduce HyTrainDNN, a heterogeneous CPU-GPU training framework, that houses the optimzer states on the CPU (FP32), a master copy of the model weights on the CPU (FP32) and a training copy of the model weights on the GPU. FWD and BWD passes are performed using Auto-Mixed Precision (by pytorch) and a Gradient Scaler (by pytorch). BWD pass is parallelyzed with the Parameter UPDATES i.e as and when the gradients for layer N is generated, they are sent to layer N-1 for BWD and to the CPU where the master copy of weights are updated. Updates happen using Deepspeed's CPU Adam (Multithreaded/ OpenMP, C++ based, Dense optimzer). 

These updates are significantly slower and hence cause idleness on the GPUs. The GPU wait for updated model weights from the CPU before they can begin FWD pass for the next iteration.

To eliminate this bottleneck at the CPU side, part of the PU are allocated to the GPU. This decision is mathematically performed by the below formulation. [Image link here]

The formula estimates k, the fraction of layers (from the start) that must be updated on the GPU (here we used Torch ADAM and not CPU Adam). We develop the basis for this formula by equating the time taken to do BWD on GPU and time to do PU for k layers on CPU with the time taken to do updates of all layers - time to update the k layers on CPU.

Note that the in the LHS, we estimate time to update k layers on GPU and the RHS, the time to update k layers is estimated in the CPU side.

It is understood that the time taken to update K layers t_upd_k_GPU is smaller than t_upd_k_CPU. We introduce a fraction alpha that is t_upd_k_GPU/ t_upd_k_CPU. We then show t_upd_k_CPU in terms of t_upd_k_GPU. 
Note that Alpha should be decided before hand by the user and can be measured by measuring the time taken to update one transformer hidden layer on GPU divided by the time taken to update the same layer on CPU.

The terms are estimated at runtime, calculated JIT followd by the splitting of optimizer states between GPU and CPU. Note that these operations happen every iteration but can be tailored to happen every few training iterations.

HyTrainDNN, maintains loss convergence despite all its memory movements. Below is the loss chart for HyTrainDNN for Llama10B model with BS Context Length, Alpha = , k (estimated at runtime) = . This experiment was run on NVIDIA A100 80GB HBM GPU and AMD EPYC CPU (HPC cluster) for a period of 3000 training iterations. 

{loss curve goes here}

Below, is shown trace diagrams generatd using VizTracer[link here], the diagram shows the GPU activity for the training iteration. Ir shows the FWD and BWD. Stream 7 shows the FWD and BWD GPU kernels while Stream 13 and 17 show Host to Device (CPU -> GPU) and Device to Host (GPU -> CPU) transfers respectively.

{table here with before image and after image, image sizes might be different}

Offloading of Optimizer states for parameter updates (was done to leverage the excess CPU memory to house optimizer states ADAM takes 4 Bytes per parameter for momentum and variance for the model which is memory intensive and occupies majority of GPU VRAM) to the CPU incurs GPU idleness leading to wasted compute cycles. 

HyTrainDNN is built on the project CoTrain (ICPP' 23)[paper link here]
Project done in affiliation with MARS Lab, CDS Department, Indian Institute of Science, Bangalore, India. Contact Karthik Namboori (me) @ karthik.namboori@fsid-iisc.in or namkarthik2003@gmail.com for collaborations, feature request or a bug report.