# 调研文档：64GB DRAM 平台运行 Qwen3.5-122B-A10B-Q4_K_M 的 MoE 内存优化方案

> 项目目标：在 64GB DRAM 平台上运行 Qwen3.5-122B-A10B-Q4_K_M，使 DRAM 使用降低 40% 以上，并在 2K-32K 长上下文中保持提示词解析速度和解码速度无明显衰减。
>
> 本文基于 `paper/` 中论文、`llama.cpp` 当前源码，以及 `airllm`、`MoE-Infinity-main`、`promoe`、`fastllm-master` 四个本地开源项目阅读整理。

## 1. 结论摘要

当前目标本质上不是单纯的量化问题，而是 **MoE 专家 working set 控制问题**。Qwen3.5-122B-A10B-Q4_K_M 的总权重规模接近 64GB DRAM 上限，长上下文还会额外消耗 KV cache、recurrent state、graph buffer、backend staging buffer 和系统内存。因此要达成 40% 以上 DRAM 下降，必须让大量非活跃专家不常驻 DRAM，而是进入 SSD/mmap/压缩冷存储，并通过路由预测、专家缓存和预取把 I/O 延迟藏到计算路径之外。

对 llama.cpp 来说，最有价值的现有基础有三点：

- `Qwen35MoE` 已经在 `src/models/qwen35moe.cpp` 中独立建模，主干层数通过 `n_layer - nextn_predict_layers` 区分 35B/122B/397B 类型，122B-A10B 对应 48 个主干层。
- MoE 核心 op 是 `GGML_OP_MUL_MAT_ID`，权重张量按 `[cols, rows, n_expert]` 存储，天然具备 expert 维度。
- `ggml-backend.cpp` 已经在调度 split 时对 `MUL_MAT_ID` 做“只拷贝本次用到专家”的优化。这说明专家级复制路径已有雏形，但仍以 host buffer 常驻为前提，尚不能把 DRAM 使用降低 40%。

推荐路线是分阶段推进：

1. 先做 **专家访问追踪 + DRAM 计量基线**，确认 2K/8K/32K prefill 和 decode 下每层实际专家集合、复用率、KV/recurrent/权重内存占比。
2. 再做 **专家权重 mmap + per-expert cache**，将路由专家从“全量常驻 host DRAM”改为“按 expert 分块驻留 + 冷数据文件映射/SSD 后备”。
3. 最后做 **预测预取 + 缓存替换 + token/expert reorder**，目标不是减少计算，而是减少 cache miss 和等待 I/O 的临界路径。

## 2. 目标拆解与核心瓶颈

### 2.1 目标拆解

目标可以拆成三个可度量指标：

- `DRAM 使用降低 40%+`：不能只看进程 RSS 峰值，还要区分权重常驻、mmap resident pages、KV cache、graph/work buffer、backend staging buffer 和系统页缓存。
- `2K-32K 提示词解析无明显衰减`：prefill 阶段一次处理大量 token，每层 Top-K 专家集合可能覆盖大部分专家，专家缓存命中率天然低于 decode。
- `解码速度无明显衰减`：decode 阶段 batch 通常较小，专家复用和跨 token 路由稳定性更强，是专家缓存/预测预取最容易生效的阶段。

### 2.2 为什么 40% DRAM 下降很难

Q4_K_M 已经把权重量化到较低 bit width，但 122B 级别模型仍接近 64GB 平台上限。MoE 稀疏激活只降低每 token 计算量，不自动降低权重常驻内存，因为默认加载时所有专家权重都在 GGUF tensor/buffer 中。

在 llama.cpp 当前结构中，Qwen3.5MoE 每层路由专家主要是：

- `ffn_gate_inp`：router 权重，体积较小，应该常驻。
- `ffn_gate_up_exps`：合并 gate/up 专家权重，体积大，按 expert 维度访问。
- `ffn_down_exps`：down 专家权重，体积大，按 expert 维度访问。
- shared expert：所有 token 始终使用，体积较小，应常驻。

因此可卸载对象主要是 `ffn_gate_up_exps` 和 `ffn_down_exps`。如果只把专家从 GPU buffer 搬回 CPU host buffer，64GB DRAM 的总占用不会降；必须让冷专家不占或少占 DRAM resident pages。

### 2.3 长上下文的额外压力

Qwen3.5MoE 是 hybrid 模型，部分层是 full attention，部分层是 recurrent/linear attention。长上下文压力分两类：

- full attention 层需要 KV cache，2K 到 32K 会线性增加。
- recurrent 层需要 recurrent state，增长方式不同，但仍需要独立内存规划。

所以专家卸载不能把所有内存预算都拿走。比较稳妥的预算策略是将 64GB 拆成：

- 常驻密集权重、router、shared experts、norm/output 等。
- KV cache/recurrent state。
- graph/work/staging buffer。
- routed expert DRAM cache。
- OS 与文件页缓存余量。

## 3. llama.cpp 现状分析

### 3.1 Qwen3.5MoE 架构入口

关键文件：

- `src/llama-arch.cpp`：注册 `LLM_ARCH_QWEN35MOE`，配置 tensor 名称和 `MUL_MAT_ID` op 分类。
- `src/models/qwen35moe.cpp`：加载 Qwen3.5MoE 超参、tensor、主图和 MTP 图。
- `src/llama-graph.cpp`：通用 `build_moe_ffn()`。
- `ggml/src/ggml.c`：定义 `ggml_mul_mat_id()`。
- `ggml/src/ggml-backend.cpp`：backend scheduler，已有按 used expert 复制的特殊路径。
- `ggml/src/ggml-vulkan/`：Vulkan 后端，包含 fused topk-moe 和 `MUL_MAT_ID` shader/pipeline。

`qwen35moe.cpp` 中的关键事实：

- `nextn_predict_layers` 表示 MTP/NextN 额外 decoder block，主干推理只跑 `n_layer - nextn_predict_layers`。
- full attention interval 默认为 4，主干层中 `(i + 1) % full_attn_interval != 0` 的层被标为 recurrent。
- 主干层和 MTP 层都创建 routed experts 与 shared experts。
- 122B-A10B 类型通过主干层数 48 识别。

### 3.2 MoE 计算图关键路径

`build_layer_ffn()` 对 Qwen3.5MoE 的封装可以概括为：

```text
attn_post_norm
  -> build_moe_ffn(
       gate_inp,
       down_exps,
       gate_up_exps,
       n_expert,
       n_expert_used,
       SOFTMAX,
       SILU/SwiGLU)
  -> shared expert FFN
  -> sigmoid(shared_gate) * shared_expert_output
  -> routed_moe_output + shared_output
```

`build_moe_ffn()` 内部的核心步骤：

1. `build_lora_mm(gate_inp, cur)` 得到 `[n_expert, n_tokens]` logits。
2. `softmax/sigmoid` 得到 routing probs。
3. `ggml_argsort_top_k()` 选出 `[n_expert_used, n_tokens]` expert ids。
4. `ggml_get_rows()` 取出选中专家权重。
5. `build_lora_mm_id(gate_up_exps, cur, selected_experts)` 做 gate/up fused expert matmul。
6. split gate/up，执行 `silu(gate) * up`。
7. `build_lora_mm_id(down_exps, cur, selected_experts)` 做 down projection。
8. 按路由权重加权并 sum 到 `[n_embd, n_tokens]`。

这里最适合插入专家缓存逻辑的点不是 shader 内部，而是 `selected_experts` 已经产生、`MUL_MAT_ID` 即将消费权重的时候。因为此时已经知道本层需要哪些 expert，可以做精确 on-demand，也可以触发下一层预取。

### 3.3 ggml/后端已有 MoE offload 基础

`ggml_mul_mat_id()` 定义了 MoE 专家矩阵乘：

```text
as  -> [cols, rows, n_expert]
b   -> [cols, n_expert_used, n_tokens]
ids -> [n_expert_used, n_tokens]
c   -> [rows, n_expert_used, n_tokens]
```

`ggml-backend.cpp` 中已有特殊处理：当输入是 weights buffer、host buffer，且当前 split 的第一个节点是 `GGML_OP_MUL_MAT_ID` 时，调度器会：

- 从 `node->src[2]` 读取 ids tensor。
- 统计本次实际使用的 expert id bitset。
- 将连续 expert id 合并成区间。
- 只把这些 expert 区间复制到目标 backend copy tensor。

这个优化对传统 GPU offload 很重要，但对本项目只有一半价值：

- 它能减少 CPU->GPU/backend copy 量。
- 它不能减少 host DRAM，因为原始权重仍在 host buffer 中。
- 它依赖当前节点就是 `MUL_MAT_ID`，对复杂 split、预取、跨层 cache 还不够。

### 3.4 Vulkan 后端现状

Vulkan 后端包含两类 MoE 相关优化：

- top-k 路由融合：`topk_moe.comp` 支持 softmax/sigmoid、bias、Top-K、权重归一化等流程。
- `MUL_MAT_ID` matmul：`mul_mm.comp`、`mul_mmq.comp`、`mul_mat_vec_base.glsl` 等通过 `MUL_MAT_ID` 宏读取 expert id，并按 expert 维度偏移访问权重。

Vulkan 的 shader 不是理想的第一修改点。原因是：

- shader 只能消费已经绑定的 buffer，不能直接做 SSD/mmap I/O。
- 真正的“专家是否已在 DRAM/backend buffer”由 host-side scheduler/cache 管。
- 如果先改 shader，很容易绕不开 buffer lifetime、descriptor、同步和页驻留问题。

更稳妥的路线是先在 ggml backend scheduler/llama model loader 层做 expert cache，再让 Vulkan 继续执行现有 `MUL_MAT_ID` kernel。

## 4. 论文调研

### 4.1 MoE-Infinity

论文：`2401.14361v3.pdf`，标题为 *MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache*。

核心思想：

- 面向个人机器、batch size 通常较小的场景。
- 观察 decode 阶段专家激活具有稀疏性和复用性。
- 通过 expert activation trace 学习当前请求的稀疏模式。
- 用 sparsity-aware expert cache 指导替换和预取。

可借鉴点：

- 把专家 cache 按 `(layer, expert)` 作为基本 key。
- prefill 与 decode 要分开统计，decode 的 cache 命中收益更稳定。
- replacement policy 需要利用 trace，而不是只靠 LRU/LFU。

落地限制：

- 论文主要解决 GPU memory 不足，将专家放在 CPU host memory；本项目目标是 DRAM 总占用下降，因此必须进一步引入 SSD/mmap 冷层。
- 开源版 MoE-Infinity 偏 HuggingFace/PyTorch，和 llama.cpp/ggml 的张量生命周期差异较大。

### 4.2 ExpertFlow

论文：`2410.17954v2.pdf`，标题为 *ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling*。

核心思想：

- 用轻量 routing path predictor 一次前向预测所有 MoE 层的专家使用。
- token scheduler 把预测路由相似的 token 分组，提高专家利用率。
- predictive expert cache 只加载需要专家，并在运行时纠正误预测。

可借鉴点：

- 2K-32K prefill 阶段可以考虑 token scheduling，因为大量 token 一起进图，专家集合可能很大，调度 token 顺序能提高每次加载专家的利用率。
- 预测不需要 100% 正确，只要能提前加载高概率专家，并对 miss 有快速 fallback。

落地限制：

- predictor 训练和集成成本高。
- llama.cpp 当前图构建方式更偏静态 batch graph，插入 token reorder 需要非常谨慎，必须保证输出 token 顺序还原和 KV/recurrent state 正确。

### 4.3 FineMoE

论文：`2502.05370v2.pdf`，标题为 *Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading*。

核心思想：

- 细粒度专家卸载，缓解 coarse-grained 方案在延迟和内存之间的冲突。
- 利用 expert selection patterns 和 prompt semantic hints 指导 prefetch/cache/offload。

可借鉴点：

- 单个 expert 仍可能太大，必要时要拆成 gate/up/down 或更细的 sub-expert/chunk。
- cache 粒度越细，越能适配 64GB DRAM 紧张场景，但调度复杂度会上升。

落地限制：

- 细粒度拆分会增加 GGUF tensor layout、loader、backend copy 和调度元数据复杂度。
- 对 llama.cpp 第一阶段而言，先按 `(layer, expert, projection)` 拆分比子矩阵级拆分更稳。

### 4.4 FlashMoE

论文：`2601.17063v1.pdf`，标题为 *FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices*。

核心思想：

- 非活跃专家放到 SSD，解决 DRAM 也不足的问题。
- 缓存替换使用轻量 ML 策略融合 recency 和 frequency。
- 目标是减少 SSD I/O，提升 cache hit rate。

可借鉴点：

- 本项目要降低 40% DRAM，FlashMoE 是最贴近目标的方向。
- 初版不必马上上 ML policy，可以先实现 LRU/LFU/priority baseline，再用 trace 数据训练或拟合简单策略。
- SSD I/O 粒度必须控制，建议 expert 文件连续布局，并对 Q4_K_M block 对齐。

落地限制：

- SSD 读延迟远高于 DRAM，需要强预取和异步 I/O。
- Windows 平台还要考虑 mmap、文件缓存、异步读、页驻留统计的差异。

### 4.5 Speculating Experts

论文：`2603.19289v1.pdf`，标题为 *Speculating Experts Accelerates Inference for Mixture-of-Experts*。

核心思想：

- 在 pre-norm MoE block 中，用当前层残差流和默认向量构造 quasi-hidden state，预测下一层专家。
- 将 CPU->GPU 专家传输与当前层计算重叠。
- 对部分模型可直接执行 speculated experts，对其他模型需要 lightweight estimator 提升命中。

可借鉴点：

- llama.cpp 的 `attn_post_norm` 之后正好接 router，可以记录当前层隐状态或 router logits 来做下一层预测。
- decode 阶段每次只有少量 token，预测下一层专家的收益很直接。

落地限制：

- speculative execution 会改变实际 router 选择时的计算路径，准确性风险较大。
- 第一阶段建议只做 speculative prefetch，不做 speculative execution。

### 4.6 Fast MoE Inference

论文：`2605.11537v1.pdf`，标题为 *Fast MoE Inference via Predictive Prefetching and Expert Replication*。

核心思想：

- 预测未来 batch 中过载专家，并动态复制专家提升并行度。
- 重点解决专家负载不均和 GPU 利用率。

可借鉴点：

- 对 32K prefill，大量 token 可能集中到热门专家，专家级并行和 token grouping 有潜力。

落地限制：

- 本项目是 64GB 单机 DRAM 约束，专家复制会增加内存，不是第一优先级。
- 可作为后续性能优化，而不是 40% DRAM 降低的主线。

### 4.7 FIRM-MoE

论文：`19813-AAAI26.ChenK-ML.pdf`，标题为 *FIRM-MoE: Fine-Grained Expert Decomposition for Resource-Adaptive MoE Inference*。

核心思想：

- 将 expert 分解为更细粒度 sub-experts，降低预测不准时加载错误专家的风险。
- 通过 multi-layer expert prediction 和 resource-adaptive pre-loading 提升鲁棒性。
- 论文摘要报告 decoding 中最高 1.5x speedup 和 2.8x memory savings。

可借鉴点：

- 如果 Qwen3.5-122B 的单 expert 粒度仍太粗，后续可以按 `gate_up/down` 或更细 chunk 做缓存。
- resource-adaptive 思路适合 64GB 平台：缓存容量要根据 KV cache 长度动态收缩。

落地限制：

- sub-expert 拆分会触及量化 block、tensor stride、GGUF layout 和 backend kernel，工程量较大。
- 应作为二阶段/三阶段优化，不建议作为第一个可运行版本。

## 5. 本地开源项目对比

| 项目 | 核心策略 | 对本项目的价值 | 主要限制 |
|---|---|---|---|
| AirLLM | 按层切分权重，逐层加载到 GPU，用完释放，可选预取 | 极简 baseline，说明“加载与计算重叠”的基本收益 | 粒度是 layer，不是 expert；对 MoE 122B 长上下文会产生巨大 I/O 压力 |
| MoE-Infinity | `(layer, expert)` cache、trace-based predictor、prefetcher、priority replacement | 专家缓存架构最直接，可借鉴 cache key、保护区、prefetch matrix | 主要减少 GPU 显存，不减少 host DRAM；PyTorch 集成方式和 llama.cpp 差异大 |
| promoe | C++ worker、cache policy、prefetch scheduler、early preempt、expert reorder、chunk prefetch | 和 llama.cpp 更接近，特别是状态机、精准任务抢占、per-layer queue | 偏 CUDA/PyTorch extension，需要移植到 ggml backend/Vulkan |
| fastllm | C++ MoE 模型中直接 TopK 后循环执行专家，支持 merged swiglu/shared expert | 便于理解 MoE 的朴素计算路径和 shared expert 合并 | 不是专家 offload/cache 主线，代码结构和 ggml graph 不同 |

### 5.1 AirLLM

AirLLM 的 `airllm_base.py` 使用 `init_empty_weights()` 构建空模型，然后按 layer shard 加载/释放。它的价值是展示最小可行卸载模型：只常驻当前计算所需权重，预取下一层以减少等待。

对本项目的启发：

- 可以先做“专家 shard 文件 + 当前层按需加载”原型，验证 DRAM 能否降下来。
- 但直接按层加载会破坏速度目标，因为 Qwen3.5MoE 每层包含很多专家，prefill 需要的专家集合大。

### 5.2 MoE-Infinity

开源代码中 `expert_cache.py` 使用 GPU/CPU 两级 cache，`ExpertPredictor` 用 tracer 找相似 trace，`ExpertPrefetcher` 根据 expert matrix 排序并 enqueue prefetch。

对本项目的启发：

- cache entry 应该包含 `layer_idx`、`expert_idx`、访问频率、时间戳、预测权重、保护状态。
- cache miss 和 prefetch miss 要分别统计。
- prefill 和 decode 的缓存策略要分开，因为 prefill 激活专家更多，decode 更适合复用预测。

### 5.3 promoe

promoe 的 C++ worker 更贴近系统实现。`cache.cpp` 提供 FIFO/LRU/static/NN/MIN 等策略，`prefetcher.cpp` 里有：

- `reorder_experts()`：按 done/current/partial/miss 重排专家访问顺序。
- `preempt_one_layer_without_reorder_()`：精确专家到来时抢占预取队列。
- `chunk_prefetch`：把一个专家拆为多个参数块任务。
- `wait_expert()`：真正计算前等待专家进入 ready/launching 状态。

对本项目的启发：

- llama.cpp 可先实现 `ExpertCacheManager` + `ExpertPrefetchQueue`，状态至少包含 `cold/loading/ready/in_use/evicting`。
- 初版可以不改变专家执行顺序，只做 on-demand + async prefetch。
- 第二阶段再引入 expert reorder，减少已经 ready 的专家等待 partial/miss 专家。

### 5.4 fastllm

fastllm 的 `src/models/moe.cpp` 朴素展示了 MoE 计算：

- router linear + softmax + TopK。
- 对每个 token 的 Top-K expert 循环执行 gate/up/down。
- shared expert 始终执行，并通过 sigmoid gate 加权。

对本项目的启发：

- fastllm 更适合理解算法路径，不适合作为 llama.cpp 的内存优化移植模板。
- 它提醒我们 shared expert 不能卸载为冷数据，否则每层每 token 都会引入固定 miss。

### 5.5 开源项目中的 SSD / Disk Offloading 方式

现有开源项目对“offloading”的定义并不完全相同，需要区分三类：

| 类型 | 代表项目 | 数据所在层级 | 加载粒度 | 对 DRAM 降低的帮助 |
|---|---|---|---|---|
| Layer disk offload | AirLLM | disk -> CPU -> GPU | 整层 shard | 能降低 GPU/部分工作集内存，但对 MoE 专家复用不够细 |
| Expert host cache | promoe | CPU pinned host -> GPU cache | `(layer, expert, param/chunk)` | 降低 GPU 显存，不降低 host DRAM 总权重常驻 |
| Tensor disk offload + AIO | MoE-Infinity | disk file -> pinned buffer/device cache | tensor id / expert tensor | 具备 SSD 冷层雏形，可借鉴索引、分区和优先级 AIO |
| SSD expert cache | FlashMoE 论文 | SSD cold pool -> DRAM expert cache -> device/backend | `(layer, expert)` 或 chunk | 最贴近 64GB DRAM 目标，需要强预取和替换策略 |

AirLLM 的磁盘卸载方式比较直接：`utils.py` 先把 HuggingFace 权重拆成 `splitted_model`，`airllm_base.py` 在 forward 时用线程池提前 `load_layer_to_cpu()`，再 `move_layer_to_device()`。这是一种 **按层流式加载**：优点是实现简单、峰值 GPU 内存低；缺点是 MoE 模型每层包含大量 routed experts，prefill 阶段会频繁读取整层，不能利用 Top-K 稀疏性。

promoe 的主要价值不是 SSD，而是专家缓存状态机。`model_loader.cpp` 把专家权重登记到 `ExpertHandler.host_data`，`cache.cpp` 维护 GPU cache slot 和 FIFO/LRU/NN/MIN 策略，`prefetcher.cpp` 实现 precise job、prefetch job、early preempt、chunk prefetch 和 expert reorder。它默认假设专家源数据已经在 CPU host/pinned memory 中，因此 **不能直接降低 DRAM 总占用**；但它的状态机非常适合迁移到 llama.cpp 的 expert cache manager。

MoE-Infinity 的开源代码更接近 SSD 冷层。`core/aio/archer_tensor_handle.cpp` 中的 `StoreTensor()` 会把 tensor 写到 `archer_param_<file_id>` 分区文件，并把 `tensor_id -> file_id/offset/size/shape/options` 写进 `archer_index`；`ReadTensor()` 再按 tensor id 读取到目标 memory pointer。`core/aio/archer_prio_aio_handle.cpp` 中使用 1MB block、I/O thread pool、high/low priority queue 和 pinned memory pool 组织异步读写。这说明它不是简单 `torch.load()`，而是有 **索引化 tensor 冷存储 + 优先级 AIO** 的设计。

不过 MoE-Infinity 论文和开源实现的主要目标仍是 GPU 显存不足，很多路径会把 dense weights 和 hot sparse experts 留在 CPU/host cache。对于本项目，MoE-Infinity 的可借鉴点是：

- 用稳定的 `tensor_id` 或 `(layer, expert, tensor_kind)` 建立 SSD 索引。
- 将冷专家文件做大文件分区，避免每个 expert 一个小文件造成文件句柄和随机 I/O 开销。
- 读写按固定 block 对齐，使用 pinned staging buffer，支持高优先级 on-demand 和低优先级 prefetch。
- cache limit 需要先扣除 dense 常驻权重，再把剩余容量分给 sparse expert cache，类似 `GetSparseCacheLimit()` 的思路。

FlashMoE 论文补齐的是 MoE-Infinity/promoe 没有完全解决的问题：当 CPU DRAM 也不够时，inactive experts 必须放到 SSD，而 cache replacement 必须减少 SSD read amplification。它的关键启发是：SSD 层不是“慢一点的内存”，而是需要被 predictor/cache policy 主动保护的冷层；否则 32K prefill 或低命中 decode 会让 I/O 成为主瓶颈。

## 6. 方案建议

### 6.1 总体架构

建议采用三级存储模型：

```text
常驻 DRAM:
  dense weights, router, norm, attention, shared experts, KV/recurrent state

专家 DRAM cache:
  routed experts: (layer, expert, gate_up/down) hot set

冷存储:
  SSD / mmap file: routed expert shards, Q4_K_M block 对齐
```

推理路径：

```text
build graph / run router
  -> 得到 selected_experts
  -> ExpertCacheManager 确认本层专家 ready
  -> miss: 同步/异步加载 cold expert 到 DRAM cache
  -> 命中: 直接绑定现有 expert slice
  -> ggml MUL_MAT_ID / Vulkan kernel 执行
  -> 根据 logits/trace 预测后续层专家并预取
```

### 6.2 专家模型卸载的 SSD 层级设计

针对 Qwen3.5-122B-A10B-Q4_K_M，建议把模型权重分成 dense 常驻区、shared expert 常驻区、routed expert SSD 冷区和 routed expert DRAM 热区：

```text
Level 0: 常驻 DRAM / backend buffer
  token embedding, attention/recurrent weights, router, norm, output,
  shared experts, KV cache, recurrent state

Level 1: Expert DRAM cache
  routed expert hot set:
    (layer, expert, gate_up)
    (layer, expert, down)
  由 cache manager 控制容量、pin、evict、in-flight 状态

Level 2: SSD / mmap cold store
  routed expert cold pool:
    按 layer -> expert -> projection 建索引
    大文件连续分区，记录 offset/size/type/quant block 对齐

Level 3: 原始 GGUF / 转换源文件
  只作为构建 expert cold store 的来源，不在推理时随机解析
```

这个层级的核心原则是：**router 和 shared expert 不卸载，routed expert 才进入 SSD 层级**。router 每层每 token 都要先运行，shared expert 每层每 token 都会执行；把它们卸载会引入固定 I/O，得不偿失。`ffn_gate_up_exps` 和 `ffn_down_exps` 才是主要冷数据。

推荐的 SSD expert key：

```text
ExpertKey = {
  layer_id,
  expert_id,
  tensor_kind: gate_up | down,
  quant_type,
  offset,
  nbytes,
  alignment
}
```

第一版可以把 `(gate_up + down)` 作为一个 expert bundle 一起读入，减少状态机复杂度；第二版再拆成 `gate_up` 和 `down` 两个 projection，因为有些预取策略可能只需要提前 gate/up，down 在激活完成后再加载。更细的 chunk/sub-expert 粒度应放到后续阶段，因为会触及 Q4_K_M block 对齐、tensor stride 和 Vulkan buffer copy。

SSD 读路径建议分成两条队列：

- `on-demand high priority`：当前层 `selected_experts` 已经确定，缺失专家必须尽快读入，计算会等待它。
- `prefetch low priority`：根据下一层或多层预测提前读入，允许被抢占、取消或降级。

每个 expert cache entry 至少需要这些状态：

```text
cold_on_ssd
loading_prefetch
loading_ondemand
ready
in_use
evicting
```

在 llama.cpp 中，最自然的接入点是 `selected_experts` 生成之后、`MUL_MAT_ID` 消费权重之前。具体做法是：

- `build_moe_ffn()` 或 callback 暴露当前层 `selected_experts`。
- `llama-expert-cache` 根据 ids 计算 used expert set。
- cache miss 时从 SSD/mmap cold store 读到 DRAM expert cache window。
- backend scheduler 继续使用现有 `MUL_MAT_ID` 路径，只是权重来源从“全量 host tensor”变成“cache window / selected expert buffer”。
- Vulkan shader 仍只消费已绑定 buffer，不直接做 SSD I/O。

需要特别注意 Windows/统一内存平台的两点：

- 如果使用 mmap，文件页缓存可能把冷专家逐步变成 resident pages，导致“看似 offload，实际 DRAM 又涨回去”。评测必须统计 resident/page cache，而不是只看虚拟地址空间。
- 64GB DRAM 下，expert cache 容量要随 2K/8K/32K KV cache 动态变化。长上下文越大，Expert DRAM cache 越要收缩，否则会挤压 KV/recurrent state。

### 6.3 第一阶段：可观测性和基线

先不改内存布局，添加统计：

- 每层每 token 的 `selected_experts`。
- prefill/decode 分开的 expert unique count。
- 每层专家复用率、热门专家分布、跨 token/top-k 重合率。
- `ggml_backend_sched_compute_splits()` 中 used expert copy 的数量和字节数。
- RSS、mmap resident、KV cache、backend buffer 峰值。

验收：

- 输出 2K/8K/32K 的专家访问热力图。
- 明确 decode 阶段 cache size 从 10%/20%/40% 专家容量时的理论 hit rate。
- 明确 prefill 阶段是否需要 token scheduling。

### 6.4 第二阶段：专家 DRAM cache

修改方向：

- 在 model loader 阶段识别 routed expert 大 tensor：`ffn_gate_up_exps`、`ffn_down_exps`。
- 保留逻辑 tensor 形状，但底层 data 指向可替换 expert cache window。
- 冷权重以 expert-contiguous 方式存储，支持 `(layer, tensor_kind, expert_id)` 定位。
- cache manager 负责加载、pin、evict、保护 in-flight expert。

推荐初版粒度：

- key: `(layer, expert)`
- value: `gate_up + down` 两个 projection 一起加载
- 常驻：router + shared expert

后续优化粒度：

- key: `(layer, expert, projection)`，把 `gate_up` 与 `down` 分开缓存。
- key: `(layer, expert, projection, chunk)`，对齐 FIRM-MoE/FineMoE 的细粒度方向。

### 6.5 第三阶段：预测预取

优先级建议：

1. Decode 阶段基于历史 trace 的 next-layer predictor。
2. Decode 阶段基于当前层 router logits/top-k 的 multi-layer predictor。
3. Prefill 阶段按 token route grouping，减少专家 thrash。
4. 长上下文下按 KV cache 余量动态调整 expert cache capacity。

不建议第一版做：

- speculative execution。
- 训练大型 transformer predictor。
- shader 内直接处理 SSD/mmap。

### 6.6 缓存替换策略

初版策略：

- `LRU`：最简单 baseline。
- `LFU`：对 decode 热点专家有效。
- `Layer-aware LRU`：优先保留接下来层更可能使用的专家。
- `Priority`：`score = alpha * recent + beta * freq + gamma * predicted_prob - delta * memory_cost`。

必须区分保护状态：

- 正在当前层精确使用的 expert 不可 eviction。
- 已经预取且即将在后续层使用的 expert 应有短期保护。
- 长上下文导致 KV cache 扩张时，expert cache 可以动态收缩。

### 6.7 对 40% DRAM 下降的现实判断

如果 routed experts 占模型权重的大头，且能把大部分冷专家从 resident DRAM 移到 SSD/mmap，那么 40% DRAM 下降有可行性。但必须满足：

- 冷专家不被一次性读入常驻 host buffer。
- 文件页缓存不把所有专家隐式驻留，需要监控 resident pages。
- cache capacity 明确受控，不能让预取无限扩大 DRAM。
- prefill 阶段 miss 不导致 2K-32K 明显变慢。

风险最高的是 prefill。decode 因为每 token 只访问 Top-K，预测预取和缓存更容易保持速度；prefill 一次处理大量 token，可能每层激活很多专家，导致“为了不慢而必须加载大量专家”，从而抵消 DRAM 节省。

## 7. 评测计划

### 7.1 指标

- `RSS_peak`、`Private Bytes`、`mmap resident`、`page cache`。
- `weights_resident_bytes`、`expert_cache_bytes`、`KV_cache_bytes`、`backend_work_bytes`。
- `prefill tok/s`：2K、8K、16K、32K。
- `decode tok/s` 或 TPOT。
- `expert_cache_hit_rate`、`prefetch_hit_rate`、`miss_wait_ms`。
- `SSD read bytes/token`、`read amplification`。
- 输出质量 sanity check：固定 prompt 的 logits/token 一致性。

### 7.2 基线

- 原始 llama.cpp：全量 Q4_K_M 常驻。
- `--no-mmap` 与 mmap 模式对比。
- 只启用现有 `MUL_MAT_ID` used expert copy 的 GPU/backend offload。
- 专家 cache 不预取，只 on-demand。
- 专家 cache + LRU/LFU。
- 专家 cache + trace/predict prefetch。

### 7.3 建议实验矩阵

| 实验 | 上下文 | cache 容量 | 预取 | 目标 |
|---|---:|---:|---|---|
| E0 | 2K | none | none | 原始性能/内存 |
| E1 | 2K | 20% routed experts | off | on-demand miss 成本 |
| E2 | 2K | 20% routed experts | on | 预取收益 |
| E3 | 8K | 20%/40% | on | prefill 专家覆盖率 |
| E4 | 32K | dynamic | on | 长上下文 cache/KV 权衡 |
| E5 | decode 512 tokens | 10%/20%/40% | on | TPOT 稳定性 |

## 8. 实施路线

### Milestone 0：阅读和测量

- 输出源码架构图和 Qwen3.5MoE 计算图。
- 给 `selected_experts` 加 tracing callback。
- 建立内存和速度基线。

### Milestone 1：只读专家访问分析

- 不改变权重布局，仅记录每层 experts。
- 生成 prefill/decode 热力图。
- 计算理论 cache hit rate。

### Milestone 2：专家 cache 原型

- 新增 `llama-expert-cache.{h,cpp}`。
- routed expert tensor 支持 per-expert host cache window。
- 冷权重从 mmap/SSD 按 expert 读取。
- 保持 `MUL_MAT_ID` 对外形状不变。

### Milestone 3：异步预取

- 在 router/top-k 后触发本层 on-demand 和后续层 prefetch。
- 引入 copy/read 线程、状态机、in-flight 保护。
- 对 miss 等待时间做 profiling。

### Milestone 4：Vulkan/backend 优化

- 让 Vulkan buffer 与 expert cache window 生命周期匹配。
- 复用现有 `MUL_MAT_ID` pipeline。
- 必要时增加按 expert 区间 copy 的 staging/persistent buffer。

### Milestone 5：长上下文优化

- KV cache 与 expert cache 动态预算。
- prefill token grouping。
- layer-aware/priority replacement。

## 9. 当前最推荐的下一步

现阶段最有价值的不是马上改 shader，而是做一个最小但可靠的观测闭环：

1. 在 `build_moe_ffn()` 的 `selected_experts` 节点或 backend compute 之后导出 ids。
2. 跑 2K、8K、32K prompt，分 prefill/decode 统计每层 expert unique count。
3. 基于统计结果模拟不同 expert cache 容量下的 hit/miss。
4. 决定第一版 cache 粒度是 `(layer, expert)` 还是 `(layer, expert, projection)`。

只要这一步数据站得住，后面的工程就不会盲打。这个项目最怕的是一上来做很漂亮的 cache manager，最后发现 32K prefill 每层几乎扫完整个 expert pool，速度目标被 I/O 击穿。
