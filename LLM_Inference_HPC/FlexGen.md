# Background
## Research Question Specification

There are two main targets of optimization for LLM: **latency-oriented** and **throughput-oriented**, while most research focus on the former.

Latency-oriented is often used in **interactive** use cases such as chatbots. And Throughput-oriented is often used in "**back of house**" tasks such as benchmarking, information extracting, data wrangling and form processing. 
**Key Points**: require running LLM inference in batches over a large number of tokens, and are less sensitive to latency.

And if divided by computing resources, the optimization can be divided into **enough resources** (the weights, KV cache and activation can be loaded all in GPU) and **inadequate resources** (For example, we only have a 16 GB GPU for GPT-175B, which needs at least 5 A100(80G) GPUs for model weights only. )

This paper focuses on the **throughput-oriented** and **inadequate resources**.

And Prior efforts to lower resource requirements of LLM inference correspond to three directions:
1) **model compression to decrease total memory footprint**
2) **collaborative inference to amortize inference cost via decentralization** (**Petals**, provides a collaborative reasoning and fine-tuning platform that lowers the hardware threshold for running large models by sharing resources among multiple users, actually it uses many GPUs).
3) **offloading to utilize memory from CPU and disk**(**DeepSpeed** and **Accelerator**)
However, Research in the first two directions often assume that the model fits into the GPU memory and thereby struggle to run 175B-scale models with a single commodity GPU

On the other hand, state-of-theart offloading-based systems in the third category do not achieve acceptable throughput on a single GPU due to inefficient I/O scheduling and tensor placement

So this paper present FlexGen, an offloading framework for high-throughput LLM inference. FlexGen aggregates memory from the GPU, CPU, and disk, and efficiently schedules I/O operations, along with possible compression methods and distributed pipeline parallelism.
# Methods and Contribution
## Methods:offloading
Main Challenge
1. efficient **offloading** strategy: there are three kinds of tensors: weights, activations, and key-value (KV) cache. The strategy should specify what tensors to offload, where to offload them within the three-level memory hierarchy, and when to offload them during inference.
2. effective **compression** strategies: Previous works have demonstrated promising results in compressing the weights and activations of LLMs. However, when combining compression with offloading for high-throughput inference, the I/O costs and memory reduction of the weights and KV cache become more important, motivating alternative compression schemes.





Motivated by the emerging demand for latency-insensitive tasks with batched processing, this paper initiates the study of high-throughput LLM inference using limited resources, such as a single commodity GPU. 

We present FlexGen, a high-throughput generation engine for running LLMs with limited GPU memory. 

FlexGen can be flexibly configured under various hardware resource constraints by aggregating memory and computation from the GPU, CPU, and disk. 

By solving a linear programming problem, it searches for efficient patterns to store and access tensors. 

FlexGen further compresses the weights and the attention cache to 4 bits with negligible accuracy loss. 

These techniques enable FlexGen to have a larger space of batch size choices and thus significantly increase maximum throughput. 

As a result, when running OPT-175B on a single 16GB GPU, FlexGen achieves significantly higher throughput compared to state-of-the-art offloading systems, reaching a generation throughput of 1 token/s for the first time with an effective batch size of 144. 

On the HELM benchmark, FlexGen can benchmark a 30B model with a 16GB GPU on 7 representative sub-scenarios in 21 hours.

$$
T=T_{pre}\cdot l+T_{gen}\cdot (n-1)\cdot l
$$
$$
T_{pre}=max(ctog^p,gtoc^p,dtoc^p,ctod^p,comp^p)
$$
$$
Similarly,~~~T_{gen}=max(ctog^p,gtoc^p,dtoc^p,ctod^p,comp^p)
$$


好的！以下是对这个部分的详细解释，以及公式的数学意义和推导过程，公式会用 **LaTeX** 格式输出。

---
# load from disk and CPU to GPU
### **1. 背景**
在推理过程中，数据（如权重、激活值、KV 缓存等）需要从磁盘加载到 GPU，这会引入 **I/O 延迟**。  
我们用 **`dtocg`** 表示从磁盘到 GPU 的 I/O 总延迟，其中涉及的数据包括：
- **权重（weights）**：模型的参数。
- **激活值（activations）**：每层中间计算结果。
- **KV 缓存（Key-Value cache）**：存储用于注意力机制的上下文信息。

这些数据的大小与模型结构、输入序列的长度和 block size（块大小）有关。

---

### **2. 数据大小计算公式**

#### **(a) 权重的大小**
Transformer 模型中每层的权重大小公式为：
$$
X_{weights}= 8h_1^2 + 4h_1 \cdot h_2
$$
- $h_1$ 是隐藏层的维度（hidden size）。
- $h_2$ 是 MLP（多层感知机）的隐藏层维度。

**解释**：
- **$8h_1^2$**：对应的是自注意力机制中的权重矩阵（Query、Key、Value 和 Output，分别为 $h_1 \times h_1$ 的矩阵，共 4 个）。
- **$4h_1 \cdot h_2$**：对应的是 MLP 层的权重，其中两部分分别是 $h_1 \times h_2$ 和 $h_2 \times h_1$ 的矩阵。

#### **(b) 激活值的大小**
每层的激活值大小公式为：
$$
X_{activation} = 2 \cdot \text{bls} \cdot h_1
$$
- $\text{bls}$ 是 block size（块大小）。
- $h_1$ 是隐藏层的维度。

**解释**：
- 激活值是每层的中间计算结果，包含前向传播和反向传播所需的值。
- 对于每个块中的 $\text{bls}$ 个 token，激活值的大小与隐藏层维度 $h_1$ 成正比。

#### **(c) KV 缓存的大小**
每层的 KV 缓存大小公式为：
$$
\text{KV Cache} = 4 \cdot \text{bls} \cdot \left(s + \frac{n}{2}\right) \cdot h_1
$$
- $\text{bls}$ 是 block size（块大小）。
- $s$ 是 prompt 的长度（即输入序列的长度）。
- $n$ 是要生成的 token 数。
- $h_1$ 是隐藏层的维度。

**解释**：
- KV 缓存存储的是注意力机制中 Key 和 Value 矩阵，用于处理长序列的上下文。
- **$s$** 是 Prompt 的长度，表示模型需要存储的上下文 token。
- **$\frac{n}{2}$** 是生成的 token 的平均长度贡献，假设解码阶段每个 token 访问了一半的缓存。

---

### **3. I/O 延迟公式**

从磁盘到 GPU 的延迟 **`dtocg`** 是由权重、激活值和 KV 缓存的加载时间组成的。  
磁盘到 CPU 的带宽用 $\text{bandwidth}_{\text{disk→cpu}}$ 表示。  
如果权重、激活值和 KV 缓存分别有 $w_d, h_d, c_d$ 的比例需要从磁盘加载，则延迟公式为：

$$
dtoc^{g} = \frac{1}{\text{bandwidth}_{\text{disk→cpu}}} \cdot \left( (8h_1^2 + 4h_1 \cdot h_2) \cdot w_d + 4 \cdot \text{bls} \cdot \left(s + \frac{n}{2}\right) \cdot h_1 \cdot c_d + 2 \cdot \text{bls} \cdot h_1 \cdot h_d \right)
$$

---

### **4. 公式解释**

#### **(a) 权重的加载延迟**
$$
\frac{(8h_1^2 + 4h_1 \cdot h_2) \cdot w_d}{\text{bandwidth}_{\text{disk→cpu}}}
$$
- $w_d$ 是从磁盘加载的权重比例。
- 权重的大小是 $8h_1^2 + 4h_1 \cdot h_2$。
- 延迟与磁盘到 CPU 的带宽成反比。

#### **(b) KV 缓存的加载延迟**
$$
\frac{4 \cdot \text{bls} \cdot \left(s + \frac{n}{2}\right) \cdot h_1 \cdot c_d}{\text{bandwidth}_{\text{disk→cpu}}}
$$
- $c_d$ 是从磁盘加载的 KV 缓存比例。
- KV 缓存的大小与序列长度 $s + \frac{n}{2}$、块大小 $\text{bls}$ 和隐藏维度 $h_1$ 成正比。

#### **(c) 激活值的加载延迟**
$$
\frac{2 \cdot \text{bls} \cdot h_1 \cdot h_d}{\text{bandwidth}_{\text{disk→cpu}}}
$$
- $h_d$ 是从磁盘加载的激活值比例。
- 激活值的大小与块大小 $\text{bls}$ 和隐藏维度 $h_1$ 成正比。

---

### **5. 总结公式**
最终，磁盘到 GPU 的 I/O 延迟（`dtocg`）公式如下：
$$
d_{\text{tocg}} = \frac{1}{\text{bandwidth}_{\text{disk→cpu}}} \cdot \left( (8h_1^2 + 4h_1 \cdot h_2) \cdot w_d + 4 \cdot \text{bls} \cdot \left(s + \frac{n}{2}\right) \cdot h_1 \cdot c_d + 2 \cdot \text{bls} \cdot h_1 \cdot h_d \right)
$$

**总结解释**：
1. 权重、激活值和 KV 缓存的大小分别由 Transformer 模型的结构（$h_1, h_2$）、序列长度（$s, n$）和 block size（$\text{bls}$）决定。
2. 每种数据的加载延迟与其大小成正比，与磁盘到 CPU 的带宽成反比。
3. 延迟是所有加载操作的总和，用 $w_d, c_d, h_d$ 表示从磁盘加载的比例。

通过优化 $w_d, c_d, h_d$ 的比例，模型可以减少磁盘的 I/O 延迟，从而提升推理效率。


以下是对这段内容的详细讲解和基于 **LaTeX** 的公式化改写，同时对重要部分进行分步分析。

---

### **1. 策略变量的定义**
一个完整的策略（policy）包含 **11 个变量**，用于描述 Transformer 模型推理过程中的调度和存储分布。变量包括：
- **块大小（block size）：$\text{bls}$**
  - 指定每个计算块的大小（如处理多少个 token）。
- **GPU 批大小（GPU batch size）：$\text{gbs}$**
  - 指定一次在 GPU 上处理的 token 的数量。
- **权重的存储位置（weight placement）：$w_g, w_c, w_d$**
  - 分别表示权重存储在 GPU、CPU 和磁盘上的比例。
- **激活值的存储位置（activation placement）：$h_g, h_c, h_d$**
  - 分别表示激活值存储在 GPU、CPU 和磁盘上的比例。
- **KV 缓存的存储位置（KV cache placement）：$c_g, c_c, c_d$**
  - 分别表示 KV 缓存存储在 GPU、CPU 和磁盘上的比例。

这些变量的值用百分比表示，例如 $w_g$ 表示存储在 GPU 上的权重比例。

---

### **2. 变量的约束**
理论上，百分比变量可以是任意实数（real number）介于 $[0, 1]$ 之间，但在实际中，张量（tensor）无法被任意分割。因此在成本模型中，我们对这些变量进行松弛处理，允许它们是任意实数，方便计算优化。

变量的约束条件包括：
- **存储总和约束**：
  - 权重存储的比例总和为 1：
    $$
    w_g + w_c + w_d = 1
    $$
  - KV 缓存存储的比例总和为 1：
    $$
    c_g + c_c + c_d = 1
    $$
  - 激活值存储的比例总和为 1：
    $$
    h_g + h_c + h_d = 1
    $$

- **硬件容量约束**：
  - GPU 内存峰值不能超过 GPU 的内存容量：
    $$
    \text{GPU peak memory} \leq \text{GPU memory capacity}
    $$
  - CPU 内存峰值不能超过 CPU 的内存容量：
    $$
    \text{CPU peak memory} \leq \text{CPU memory capacity}
    $$
  - 磁盘内存峰值不能超过磁盘的内存容量：
    $$
    \text{Disk peak memory} \leq \text{Disk memory capacity}
    $$

---

### **3. 两级优化问题的解法**

#### **第一层：枚举 $(\text{bls}, \text{gbs})$**
由于 **块大小（bls）** 和 **GPU 批大小（gbs）** 的选择范围有限，可以通过枚举来穷举可能组合：
- $\text{gbs}$ 通常是 4 的倍数。
- $\text{bls}$ 通常小于 20。
$$
gbs \% 4=0
$$
$$
bls <= 20
$$

因此，$(\text{bls}, \text{gbs})$ 的组合数量有限，枚举成本低。

#### **第二层：线性规划问题**
在固定的 $(\text{bls}, \text{gbs})$ 下，寻找最优的存储分布策略 $p = (w_g, w_c, w_d, c_g, c_c, c_d, h_g, h_c, h_d)$，可以用以下线性规划问题来求解：

目标函数：
$$
\min_p \frac{T}{\text{bls}}
$$
- $T$ 是推理计算的总延迟，随着存储策略 $p$ 的变化而变化。
- $\text{bls}$ 是块大小，目标是最小化每个块的延迟。

约束条件：
1. **硬件容量约束**：
   $$
   \text{GPU peak memory} \leq \text{GPU memory capacity}
   $$
   $$
   \text{CPU peak memory} \leq \text{CPU memory capacity}
   $$
   $$
   \text{Disk peak memory} \leq \text{Disk memory capacity}
   $$
2. **存储总和约束**：
   $$
   w_g + w_c + w_d = 1
   $$
   $$
   c_g + c_c + c_d = 1
   $$
   $$
   h_g + h_c + h_d = 1
   $$

---

### **4. 线性规划问题的快速求解**
由于线性规划问题只有 **9 个变量**（$w_g, w_c, w_d, c_g, c_c, c_d, h_g, h_c, h_d$），可以在短时间内快速求解。求解得到的存储策略 $p$ 是一个初步的存储分布方案。

---

### **5. 成本模型的实际使用**

#### **(a) 硬件参数的拟合**
在使用成本模型时，需要对硬件进行 **profiling（性能采样）**，以采集硬件性能数据点并拟合硬件参数，如：
- GPU、CPU、磁盘之间的传输带宽。
- 各种 I/O 操作的延迟。

#### **(b) 优化器的使用**
- 在拟合完硬件参数后，调用优化器可以得到一个初步的 **offloading 策略**，即存储分布方案 $p$。
- 由于松弛处理和实际硬件的复杂性（如内存碎片化问题），有时策略可能会导致内存溢出。

#### **(c) 手动调整**
- 如果初步策略导致内存不足，则需要对策略进行手动微调。
- 初步策略通常已经接近最佳，但手动调整可以进一步改进性能。

---

### **6. 总结公式化写法**

最终的优化问题可以写为：
目标函数：
$$
\min_p \frac{T}{\text{bls}}
$$

约束条件：
1. 硬件容量限制：
   $$
   \text{GPU peak memory} \leq \text{GPU memory capacity}
   $$
   $$
   \text{CPU peak memory} \leq \text{CPU memory capacity}
   $$
   $$
   \text{Disk peak memory} \leq \text{Disk memory capacity}
   $$

2. 存储总和约束：
   $$
   w_g + w_c + w_d = 1
   $$
   $$
   c_g + c_c + c_d = 1
   $$
   $$
   h_g + h_c + h_d = 1
   $$

---

### **7. 总结**

通过两级优化：
1. **第一层：枚举 $(\text{bls}, \text{gbs})$**，找到可能的块大小和 GPU 批大小组合。
2. **第二层：线性规划**，在固定 $(\text{bls}, \text{gbs})$ 下，求解最优存储策略 $p$。

这种方法可以快速生成一个合理的存储分布策略，并通过手动调整进一步优化性能，确保策略在硬件约束下高效运行。