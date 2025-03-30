# efficient-dl-systems

This repo contains implementations of some popular approaches for  efficient deep learning training and inference:

- [batching.py](batching.py) is implementation of different ways to batching dataset for LLM training and inference from simple all batches padding to [FlashAttention](https://github.com/Dao-AILab/flash-attention) method.

- [scaler.py](scaler.py) is implementation of scaler for mixed precision training similar to [PyTorch GradScaler](https://pytorch.org/docs/stable/amp.html).

- [profiler.py](profiler.py) is implementation of simple profiler that exports data in the perfetto trace event format and has PyTorch interface.

- [sync_batchnorm.py](sync_batchnorm.py) implementation of [PyTorch SyncBatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html). It is BatchNorm with AllReduce for aggregation statistics from all GPUs.

- [offloading_linear.py](offloading_linear.py) is implementation of linear layer with CPU offloading with prefetching.

- [tensor_parallel_llama.py](tensor_parallel_llama.py) is implentation LLaMa with [tensor parallelism](https://huggingface.co/docs/text-generation-inference/en/conceptual/tensor_parallelism). As bonus each linear layers for TP has gradient checkpointing. Also this file contains TP LLaMa with [PyTorch DTensor](https://pytorch.org/docs/stable/distributed.tensor.parallel.html).

- [fsdp.py](fsdp.py) is simple implementation of [Fully Sharded Data Parallel](https://pytorch.org/docs/main/fsdp.html).

- [speculative_decoding.py](speculative_decoding.py) is simple implementation of [SpecDec](https://arxiv.org/abs/2211.17192) for optimizing LLM inference.

- [w8a8_matrix_mul.py](w8a8_matrix_mul.py) is contains triton kernels for quantization to int8 and matrix multiplication with dequantization (also in int8).