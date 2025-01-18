# Background
## Research Question Specification

There are two main targets of optimization for LLM: **latency-oriented** and **throughput-oriented**, while most research focus on the former.

Latency-oriented is often used in interactive use cases such as chatbots. And Throughput-oriented is often used in "back of house" tasks such as benchmarking, information extracting, data wrangling and form processing. 
**Key Points**: require running LLM inference in batches over a large number of tokens, and are less sensitive to latency.

And if divided by computing resources, the optimization can be divided into **enough resources** (the weights, KV cache and activation can be loaded all in GPU) and **inadequate resources** (For example, we only have a 16 GB GPU for GPT-175B, which needs at least 5 A100(80G) GPUs for model weights only. )

This paper focuses on the **throughput-oriented** and **inadequate resources**.

And Prior efforts to lower resource requirements of LLM inference correspond to three directions:
1) model compression to decrease total memory footprint
2) collaborative inference to amortize inference cost via decentralization (**Petals**, provides a collaborative reasoning and fine-tuning platform that lowers the hardware threshold for running large models by sharing resources among multiple users, actually it uses many GPUs).
3) offloading to utilize memory from CPU and disk(**DeepSpeed** and **Accelerator**)
However, Research in the first two directions often assume that the model fits into the GPU memory and thereby struggle to run 175B-scale models with a single commodity GPU

On the other hand, state-of-theart offloading-based systems in the third category do not achieve acceptable throughput on a single GPU due to inefficient I/O scheduling and tensor placement

So this paper present FlexGen, an offloading framework for high-throughput LLM inference. FlexGen aggregates memory from the GPU, CPU, and disk, and efficiently schedules I/O operations, along with possible compression methods and distributed pipeline parallelism.
# Methods and Contribution
## Methods:offloading
Main Challenge
1. efficient **offloading** strategy: there are three kinds of tensors: weights, activations, and key-value (KV) cache. The strategy should specify what tensors to offload, where to offload them within the three-level memory hierarchy, and when to offload them during inference.
2. effective **compression** strategies: Previous works have demonstrated promising results in compressing the weights and activations of LLMs. However, when combining compression with offloading for high-throughput inference, the I/O costs and memory reduction of the weights and KV cache become more important, motivating alternative compression schemes.



