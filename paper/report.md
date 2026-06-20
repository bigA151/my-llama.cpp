# llama.cpp 代码架构与 MoE 模型调研报告

> 目标：在 64GB DRAM 平台上运行 Qwen3.5-122B-A10B-Q4_K_M
>
> • DRAM 使用降低 40% 以上
> • 2K-32K 长上下文下，提示词解析速度和解码速度无明显衰减

---

## 核心困难一览

要在 64GB DRAM 上运行 70GB+ 的 MoE 模型，面临以下 **8 个核心困难**，分两类：

### A. llama.cpp 架构层面的 5 个困难

| # | 困难 | 根源 | 影响 |
|---|------|------|------|
| D1 | **Tiered Memory 抽象缺失** | `ggml_backend_buffer_usage` 只有 WEIGHTS / COMPUTE 两种，无 DRAM/SSD tier 区分 | 无法表达"张量可被逐出到 SSD，稍后再加载" |
| D2 | **tensor->data 指针直接解引用** | 所有 ggml op 实现在计算时直接访问 `tensor->data` | 数据若在 SSD 上触发 page fault，5-50ms 不可控延迟 |
| D3 | **MoE 权重存储需求超 DRAM 容量** | Qwen3.5-122B A10B Q4_K_M ≈ 70GB > 64GB DRAM | 即便使用 Q4_K_M 量化，总权重仍超出可用 DRAM |
| D4 | **Windows 下 unmap_fragment 空实现** | `llama-mmap.cpp:580` 在 Windows 上为 stub | 页面一旦读入 DRAM，进程结束前无法释放 → 方案直接不可行 |
| D5 | **缺乏 MoE-aware 预取/逐出策略** | 路由信息（`selected_experts`）未传递到内存管理层 | 无法根据路由决策预取下一层专家，无法在层计算完成后回收页面 |

### B. 三种粒度的不匹配问题（设计层面的根本矛盾）

| # | 粒度冲突 | 矛盾所在 |
|---|---------|---------|
| G1 | **卸载粒度 vs Expert 粒度（粗 vs 细）** | 卸载粒度是整个 tensor（~2.9 GB/层/单矩阵），无法只卸载不需要的 36 个专家保留活跃的 4 个 |
| G2 | **Expert 粒度 vs SSD 读取粒度（大 vs 小）** | 单个 expert ~73 MB，但 SSD 最优 I/O 为 4KB~128KB → 读取一个 expert 需约 18,672 次缺页中断 |
| G3 | **三者三角矛盾** | 卸载粒度太粗无法精细选择，Expert 粒度太大无法高效随机读，SSD 读取粒度太细需大量页面才能凑齐一个 expert |

> **核心结论**：这 8 个困难并非独立存在，而是相互叠加——llama.cpp 架构的 5 个困难（D1-D5）决定了即便解决了粒度问题，仍需在架构层面做根本性改动；G1-G3 的粒度矛盾则决定了改动方案必须以 **Expert 粒度** 作为统一的调度单元。

---

## 一、整体代码架构（三层倒三角结构）

```
┌─────────────────────────────────────────────────┐
│            第 1 层：llama API 层                  │
│  llama.h / llama.cpp                             │
│  模型加载、推理上下文、采样器、KV cache 管理        │
├─────────────────────────────────────────────────┤
│            第 2 层：模型实现层                      │
│  src/llama-*.cpp                                 │
│  架构定义、超参数、计算图构建、GGUF 解析、mmap     │
├─────────────────────────────────────────────────┤
│            第 3 层：ggml 张量库                    │
│  ggml/include/ + ggml/src/                       │
│  计算图(cgraph)、张量操作(op)、backend 抽象        │
├─────────────────────────────────────────────────┤
│    ggml-cpu    ggml-vulkan    ggml-cuda  ...     │
│    (CPU 后端)  (GPU 后端)     (GPU 后端)          │
└─────────────────────────────────────────────────┘
```

### 第 1 层：llama API 层

对外暴露纯 C API，提供模型加载、推理、采样等接口。核心结构体：

| 结构体 | 职责 |
|--------|------|
| `llama_model` | 模型级抽象，持有 hparams、所有权重张量、backend 缓冲区 |
| `llama_context` | 推理上下文，持有 KV cache、computation graph、batch 数据 |
| `llama_sampler` | 采样器，用于 token 生成时的概率采样 |

启动与推理流程：

```
llama_model_load_from_file()
  └→ llama_model_loader: 解析 GGUF 文件，初始化 mmap 映射
     └→ llama_model_create(): 创建所有张量结构
        └→ llama_init_from_model(): 创建 backend scheduler、KV cache
           └→ llama_decode(): 构建计算图 → backend scheduler 执行 → 返回 logits
```

### 第 2 层：模型实现层

| 文件 | 职责 |
|------|------|
| `llama-arch.cpp` | 定义所有支持的架构枚举（LLM_ARCH_QWEN35、LLM_ARCH_QWEN35MOE 等） |
| `llama-hparams.cpp` / `.h` | 模型超参数结构体定义（n_layer、n_expert、n_embd 等） |
| `llama-impl.cpp` | 各架构的 tensor layout 创建和算子组合 |
| `llama-graph.cpp` | **计算图构建**：对每层输出调用 ggml_* API 构建完整计算图 |
| `llama-model.cpp` | 模型类的 CRUD 操作 |
| `llama-model-loader.cpp` | GGUF 解析与张量数据加载（mmap 初始化、张量分配） |
| `llama-mmap.cpp` | mmap / mlock 的内存映射实现 |
| `llama-context.cpp` | 推理上下文管理 |
| `llama-kv-cache.cpp` | KV cache 管理 |
| `llama-sampler.cpp` | 采样器实现 |

### 第 3 层：ggml 张量库

| 组件 | 路径 | 职责 |
|------|------|------|
| ggml.h + ggml.cpp | ggml/include/ + ggml/src/ | 计算图（cgraph）、张量操作（mul_mat/add/rms_norm 等）、量化 |
| ggml-backend.h + .cpp | ggml/include/ + ggml/src/ | Backend 抽象：ggml_backend（执行流）、ggml_backend_buffer（内存）、ggml_backend_dev（设备） |
| ggml-backend-meta.cpp | ggml/src/ | 元后端：多设备张量并行（tensor parallelism） |
| ggml-vulkan | ggml/src/ggml-vulkan/ | Vulkan GPU 后端 |
| ggml-cpu | ggml/src/ggml-cpu/ | CPU 后端（含 x86/ARM SIMD 优化） |
| ggml-cuda | ggml/src/ggml-cuda/ | CUDA GPU 后端 |

#### ggml 计算图机制

ggml 的计算图是**延迟执行**的：调用 ggml_mul_mat(ctx, a, b) 不会立即计算，而是创建一个带有 GGML_OP_MUL_MAT 操作的张量节点，挂载 src[0]=a, src[1]=b 依赖关系。通过 ggml_build_forward_expand(gf, result) 将节点加入 cgraph。真正执行时，ggml_backend_sched 按拓扑序遍历所有节点，对每个节点调用对应 backend 的 op 实现。

```c
// ggml_tensor 核心结构（ggml.h:671）
struct ggml_tensor {
    enum ggml_type type;                    // 数据类型（F32/F16/Q4_K_M 等）
    struct ggml_backend_buffer * buffer;    // 后端缓冲区
    int64_t ne[GGML_MAX_DIMS];              // 各维度元素数量
    size_t  nb[GGML_MAX_DIMS];              // 各维度 stride（字节数）
    enum ggml_op op;                        // 当前张量对应的操作
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;
    struct ggml_tensor * src[GGML_MAX_SRC]; // 输入张量
    struct ggml_tensor * view_src;          // view 的源张量
    size_t               view_offs;         // view 相对于源张量的字节偏移
    void * data;                            // 数据指针
    char name[GGML_MAX_NAME];               // 张量名称（最大 64 字节）
    void * extra;                           // 后端私有数据（如 CUDA 指针）
    char padding[8];                        // 对齐填充
};
```

#### Backend 抽象层关键接口

```c
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,    // CPU 设备（系统内存）
    GGML_BACKEND_DEVICE_TYPE_GPU,    // GPU 设备（专用显存）
    GGML_BACKEND_DEVICE_TYPE_IGPU,   // 集成 GPU（共享内存）
    GGML_BACKEND_DEVICE_TYPE_ACCEL,  // 加速器（BLAS/AMX）
    GGML_BACKEND_DEVICE_TYPE_META,   // 元设备（多设备张量并行）
};

struct ggml_backend_dev_props {
    const char * name;
    size_t memory_free, memory_total;
    enum ggml_backend_dev_type type;
    struct ggml_backend_dev_caps caps;
};

enum ggml_backend_buffer_usage {
    GGML_BACKEND_BUFFER_USAGE_ANY = 0,
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,  // 权重（只读）
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,  // 计算中间结果
};
```

---

## 二、MoE 模型计算图构建（以 Qwen3.5MoE 为例）

Qwen3.5MoE 的标识：

```cpp
// llama-arch.cpp:42-43
{ LLM_ARCH_QWEN35,    "qwen35"    },
{ LLM_ARCH_QWEN35MOE, "qwen35moe" },
```

它支持 rollback（llm_arch_supports_rs_rollback 返回 true），意味着支持 KV cache 的 rollback 操作。

### 核心张量命名规则

从 LLM_TENSOR_NAMES 映射表提取的 Qwen3.5MoE 相关张量：

| 张量名 | 含义 | 典型形状 |
|--------|------|----------|
| token_embd | Token 嵌入表 | [n_embd, n_vocab] |
| output_norm | 输出层 RMSNorm | [n_embd] |
| output | LM Head 权重（可绑定 token_embd） | [n_embd, vocab_size] |
| blk.%d.attn_norm | 注意力层前 RMSNorm | [n_embd] |
| blk.%d.attn_q | Q 投影 | [n_embd, n_head×head_dim] |
| blk.%d.attn_k | K 投影 | [n_embd, n_kv_head×head_dim] |
| blk.%d.attn_v | V 投影 | [n_embd, n_kv_head×head_dim] |
| blk.%d.attn_output | Attention 输出投影 | [n_head×head_dim, n_embd] |
| blk.%d.ffn_norm | FFN 层前 RMSNorm | [n_embd] |
| blk.%d.ffn_gate_inp | 路由（gating）权重 | [n_embd, n_expert] |
| blk.%d.ffn_gate_exps | 门控专家权重 | [n_embd, n_ff, n_expert] |
| blk.%d.ffn_up_exps | Up 专家权重 | [n_embd, n_ff, n_expert] |
| blk.%d.ffn_down_exps | Down 专家权重 | [n_ff, n_embd, n_expert] |
| blk.%d.ffn_gate_up_exps | 融合 Gate+Up 专家权重 | [n_embd, n_ff*2, n_expert] |

### MoE 计算图构建（build_moe_ffn，18 个步骤）

入口：llama-graph.cpp:1348，完整实现超过 500 行，带详尽中文注释。

```
 ┌─────────────────────────────────────────────────────────┐
 │  步骤 1: Gate 计算（路由）                                │
 │  cur [n_tokens, n_embd] × gate_inp [n_embd, n_expert]^T │
 │  → logits [n_expert, n_tokens]                          │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 2: 激活函数（softmax / sigmoid / softmax_weight）   │
 │  → probs [n_expert, n_tokens]                           │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 3: 专家选择偏置（DeepSeek V3 风格）                  │
 │  selection_probs = probs + exp_probs_b                  │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 4-5: 专家分组选择（如 n_expert_groups > 1）          │
 │  先选组 → 组内 Top-K                                     │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 6-9: Top-K 选择 → weights + selected_experts       │
 │  ggml_top_k(selection_probs, n_expert_used)             │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 10-11: 权重归一化 / 缩放 / 应用                      │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 12: FFN 专家计算（核心）                             │
 │                                                          │
 │  路径 A（gate_up_exps 融合）：                             │
 │    build_lora_mm_id(gate_up_exps, cur, selected_experts) │
 │    → [n_ff*2, n_expert_used, n_tokens]                  │
 │    → 拆分为 gate [n_ff] + up [n_ff]                     │
 │                                                          │
 │  路径 B（分离投影）：                                      │
 │    up   = build_lora_mm_id(up_exps,   cur, selected)     │
 │    gate = build_lora_mm_id(gate_exps, cur, selected)     │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 13: 激活函数（SwiGLU / GeGLU / ReLU）               │
 │  SwiGLU: cur = silu(gate) × up                          │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 14: Down 投影                                      │
 │  build_lora_mm_id(down_exps, cur, selected_experts)      │
 │  → [n_embd, n_expert_used, n_tokens]                    │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 15-16: 加权聚合                                    │
 │  experts × weights → 各专家结果按列累加                   │
 │  moe_out = Σ(cur_experts[i] × weights[i])               │
 ├─────────────────────────────────────────────────────────┤
 │  步骤 17-18: 输出 [n_tokens, n_embd]                     │
 └─────────────────────────────────────────────────────────┘
```

#### 关键函数：build_lora_mm_id

这是 MoE 计算图的核心，执行**带索引的矩阵乘法**（对应 ggml 的 GGML_OP_MUL_MAT_ID）：

```cpp
// up_exps 形状: [n_embd, n_ff, n_expert]
// cur     形状: [n_embd, n_tokens]
// selected_experts: [n_expert_used, n_tokens]
// 结果   形状: [n_ff, n_expert_used, n_tokens]
up = build_lora_mm_id(up_exps, cur, selected_experts);
```

它的语义是：对每个 token，根据 selected_experts 索引，只从 up_exps 中取出对应专家的权重进行计算，而不是计算所有专家的结果再选取。这**大幅减少了计算量**（从 O(n_expert) 降为 O(n_expert_used)）。

---

## 三、SSD 层级权重卸载的困难和问题

### 3.1 当前架构的内存管理现状

#### mmap 实现细节

底层实现在 src/llama-mmap.cpp：

```cpp
// Linux/macOS 实现
addr = mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0);
if (prefetch > 0) {
    posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED);
}
```

```cpp
// Windows 实现
HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
if (prefetch > 0) {
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}
```

核心语义：**mmap 时仅建立虚拟地址映射，不读入数据**。首次访问某页面时触发 page fault，OS 从文件按需加载。

#### 模型加载过程中 mmap 的使用

llama-model-loader.cpp 中张量数据的放置策略：

```cpp
// 1. 初始化 mmap 映射（init_mappings）
if (use_mmap) {
    for (auto & file : files) {
        mapping = new llama_mmap(file.get(), prefetch, is_numa);
        mappings.emplace_back(mapping);
    }
}

// 2. 每个张量创建时，data 指针直接指向 mmap 地址
if (use_mmap) {
    cur->data = (uint8_t *)mapping->addr() + w.offs;
    // 然后通过 ggml_backend_tensor_alloc() 将 tensor 注册到目标 backend 的 buffer
}
```

对于需要 offload 到 GPU 的张量：

```cpp
ggml_backend_tensor_alloc(buf_mmap, cur, data);
// buf_mmap 其实是 GPU backend 的 buffer
// data 指向 mmap 地址
// 内部实现：将 data 从 host 拷贝到 GPU
```

生命周期末尾，unmap_fragment 释放未使用的首尾区域（但**推理过程中不做页面级管理**）：

```cpp
// llama-model-loader.cpp:1670
if (use_mmap) {
    mapping->unmap_fragment(0, mmap_used.first);
    mapping->unmap_fragment(mmap_used.second, mapping->size());
}
```

### 3.2 核心困难点

#### 困难 1：Tiered Memory 抽象缺失

当前 ggml_backend 只有两种存储类：
- GGML_BACKEND_BUFFER_USAGE_WEIGHTS（权重，只读）
- GGML_BACKEND_BUFFER_USAGE_COMPUTE（计算中间结果）

**没有** "DRAM tier" vs "SSD tier" 的区分。所有权重要么在 host buffer（DRAM），要么在 device buffer（VRAM）。系统无法表达"这个张量在推理过程中可以被逐出到 SSD，稍后再加载回来"。

#### 困难 2：tensor->data 指针的直接解引用

推理时，所有 ggml op 实现都直接解引用 tensor->data 指针。如果数据已在 DRAM 中则正常；如果数据在 SSD 上，则触发 OS 缺页中断，导致 5-50ms 的不可控延迟。MoE 的随机专家访问使得这个问题更严重——无法预知哪些专家将被选中的情况下，当前页面可能随时被选中访问。

#### 困难 3：MoE 权重的巨大存储需求

Qwen3.5-122B-A10B 的 MoE 结构：

```
假设：
  n_expert = 40
  n_embd = 8K, n_ff = 24K
  Q4_K_M 每元素约 0.5 字节

每层专家权重（约 3 个矩阵）：
  up_exps:   40 × 24K × 8K × 0.5B =   3.8GB
  gate_exps: 40 × 24K × 8K × 0.5B =   3.8GB
  down_exps: 40 × 8K × 24K × 0.5B  =   3.8GB

每层合计 ≈ 11.5GB/层 × 60 层 ≈ 690GB（未量化 FP16）
                              ≈ 70GB（Q4_K_M 量化后）
```

即便使用 Q4_K_M 量化，总权重也约 70GB，超过 64GB DRAM 的容量。

#### 困难 4：Windows 下 unmap_fragment 缺失

```cpp
// llama-mmap.cpp:580 (Windows)
void unmap_fragment(size_t first, size_t last) {
    GGML_UNUSED(first);
    GGML_UNUSED(last);
    // 空实现！Windows 下无法释放部分映射
}
```

这意味着 **Windows 平台上 mmap 的页面一旦被读入 DRAM，在进程结束前永远无法释放**。对 64GB DRAM + 70GB 模型的场景，这直接导致方案不可行。

#### 困难 5：缺乏 MoE-aware 的预取/逐出策略

当前没有任何代码来：
- **预取**：在当前层计算时，从 SSD 预取下一层可能被选中的专家
- **逐出**：某层计算完成后，将其专家权重页面标记为可回收
- **工作集管理**：根据可用 DRAM 动态调整在 DRAM 中的专家数量

虽然 build_moe_ffn 的路由步骤已经计算出每个 token 选择哪些专家，但这个信息**没有传递到内存管理层**。

### 3.3 MoE 模型特有的机会

MoE 模型的稀疏激活特性也带来独特优势，可以针对性利用：

| 特性 | 优势 |
|------|------|
| 每层只激活 Top-K 专家（A10B → 10B 活跃参数） | 每层计算只需要少量专家的权重，可以按需从 SSD 加载 |
| 路由结果可以预知下一层的专家选择 | 当前层 gate_inp 计算出路由概率后，可以预知下一层哪些专家最可能被选 |
| 专家权重的访问模式是重复的、确定的 | 同一专家可能被多个 token 选多次，适合批量读取 |

### 3.4 架构建议

要在 64GB DRAM 上运行 70GB+ MoE 模型，需要的不只是"优化"，而是架构级扩展：

**1. ggml backend 层新增 TieredMemory buffer type**

```c
// 新增 buffer usage 类型
GGML_BACKEND_BUFFER_USAGE_SWAPPABLE = 3

// 新增 on-demand load 接口
ggml_backend_tensor_load_async(tensor, priority);
ggml_backend_tensor_evict(tensor);
```

**2. MoE 专家权重 SSD 调度器**

介于 model_loader 和 backend_sched 之间的新组件，职责：
- 在 build_moe_ffn 开始前：根据路由概率预测下一层的热门专家
- 对热门专家：发起异步 SSD → DRAM 加载
- 对已完成计算的专家：标记 DRAM 页面可回收
- 维持工作集在 DRAM 容量限制内

**3. 异步 I/O pipeline**

```
┌─ Layer N 计算 ──────┐   ┌─ Layer N+1 预取 ────┐
│                      │   │                      │
│  Attn computation    │   │ 从 SSD 加载热门专家   │
│  MoE FFN computation │   │ 权重到 pin-memory    │
│  (使用已加载的专家)   │   │                      │
└──────────────────────┘   └──────────────────────┘
        时间线 ───────────────────────────────────────────→
```

**4. 针对 Windows 的替代方案**

鉴于 unmap_fragment 在 Windows 上为空实现，需要使用替代技术来实现页面级管理：
- 使用 VirtualFree(MEM_DECOMMIT) 释放物理页面，或
- 使用多个较小的 MapViewOfFile 分片，在切换时销毁旧的映射

---

## 四、mmap、Swap 和虚拟内存的区别

### 虚拟内存（Virtual Memory）

| 维度 | 说明 |
|------|------|
| 定义 | 操作系统提供的内存抽象层，每个进程有独立的连续虚拟地址空间 |
| 映射方式 | CPU MMU 通过页表将虚拟地址映射到物理地址 |
| 在 llama.cpp 中 | 所有 tensor->data 和 mapping->addr() 返回的都是虚拟地址 |
| 容量 | 64 位系统可达 128TB，远大于物理 DRAM（64GB） |
| 关键特点 | 70GB 的 mmap 文件只需要 70GB 虚拟地址空间，物理内存只在真正访问时按需分配 |

### mmap（Memory-Mapped File，内存映射文件）

| 维度 | 说明 |
|------|------|
| 定义 | 将文件直接映射到进程虚拟地址空间，对文件的访问就像读写内存 |
| 使用场景 | llama.cpp 默认使用 mmap 加载 GGUF 模型文件 |
| 加载时行为 | mmap 不读入数据，仅建立映射。首次访问触发 page fault，OS 从文件加载 |
| 推理时行为 | 每层计算时按需触发 page fault，OS 从 SSD 读入权重页面 |
| 页面释放 | Linux 可在使用后调用 madvise(MADV_DONTNEED) 释放物理页；Windows UnmapViewOfFile 只能整体解除映射 |
| 粒度控制 | 高。可指定映射范围、访问权限、预取策略 |
| 优势 | 零拷贝：无需调用 read()，避免用户态→内核态的数据拷贝；OS 自动管理 page cache |
| 劣势 | 不提供"逐出"能力，不能精细控制哪些页面留在物理内存中 |

### Swap（交换空间）

| 维度 | 说明 |
|------|------|
| 定义 | OS 将不活跃的物理内存页面换出到磁盘 swap 分区/文件，释放物理内存 |
| 管理主体 | OS kernel，完全自动（kswapd 在内存压力下触发） |
| 粒度 | 页面级（4KB / 2MB） |
| 延迟 | 触发 swap-in 时 5-50ms（取决于 SSD 性能） |
| 可控性 | 低。只能通过 vm.swappiness 等 sysctl 参数全局调优 |
| 不适合 MoE 的原因 | 1) 不可控：Kernel 决定换出哪些页面，无法保证路由频繁选中的专家留在内存<br>2) 不可预测：推理时随机的页面换入延迟导致解码速度抖动<br>3) 全局策略：影响整个系统进程，不仅是模型推理进程 |

### 三者对比总结

| 维度 | 虚拟内存 | mmap | Swap |
|------|----------|------|------|
| 管理主体 | CPU MMU（硬件） | 应用程序 + OS | OS kernel（kswapd） |
| 粒度 | 页面（4KB-2MB） | 页面 | 页面 |
| 数据来源 | 物理 DRAM | 文件（GGUF 权重文件） | 磁盘 swap 区域 |
| 初始化开销 | 无（仅映射页表） | 无（仅建立 vma） | 无（仅标记 dirty） |
| 首次访问延迟 | TLB hit: 纳秒级，TLB miss: 百纳秒级 | 触发 page fault: 毫秒级 | 触发 swap-in: 5-50ms |
| 后续访问延迟 | 与常规内存相同 | 与常规内存相同（OS page cache） | 同 mmap（在 RAM 中） |
| 可控性 | 硬件自动 | 高（madvise/mlock/munmap） | 低（仅 swappiness 等 sysctl） |
| llama.cpp 用法 | 所有张量通过虚拟地址访问 | 模型权重默认通过 mmap 加载（use_mmap=true） | 未主动使用，但 OS 可能在内存压力下被动触发 |

### 在 64GB DRAM + 70GB 模型场景下的分析

| 方案 | 可行性 | 问题 |
|------|--------|------|
| 纯 mmap（当前方案） | ✗ | 一旦触发 page fault 读入页面，就永久占用物理内存。70GB > 64GB，会导致 OOM 或大量 swap |
| mmap + OS swap | ✗ | OS swap 无法理解 MoE 的路由模式。路由选择被 swap 出的专家时产生 50ms+ 延迟，解码速度退化严重 |
| mmap + 显式 madvise(DONTNEED) | △（Linux only） | 需要应用层管理——在推理循环中，对用完的专家页面调用 DONTNEED。Windows 不支持部分 unmap，需要改用分片映射 |
| 应用层 SSD 调度器 | 推荐方案 | 这是正确的架构方向：新增一个 MoE 专家权重管理器，在 ggml backend 和模型实现之间做 tiered memory 调度 |

---

## 五、总结与建议

llama.cpp 当前架构的核心限制在于 **ggml_backend 的内存模型是二元的（host/device），没有分级存储（tiered memory）抽象**。mmap 提供的是文件→内存的零拷贝加载通路，而不是内存↔SSD 的双向交换通路。

要在 64GB DRAM 上运行 70GB+ 的 MoE 模型，**必须**在以下三个层面做出架构级扩展：

1. **ggml backend 层**：引入可标记为"可逐出"的 buffer type，支持张量在推理过程中从 DRAM 释放到 SSD
2. **模型加载层**：MoE 专家权重不直接 mmap 到 tensor->data，而是通过一个 MoE-aware 调度器按需加载/释放
3. **推理运行时**：在 build_moe_ffn 中，路由信息（selected_experts）需要回流到内存管理层，用于预取和逐出决策

这些改动是非平凡的，但 MoE 模型的稀疏激活特性（A10B → 每层只需要少部分专家权重参与计算）使得这个方向在理论上可行。核心挑战在于工程实现的质量：I/O 流水线的延迟隐藏、权重调度策略的准确性、以及跨平台（特别是 Windows）的兼容性。

---

*报告生成于 2026-06-16，基于 llama.cpp 项目源码结构分析。*

## 六、三种粒度及其对 SSD 卸载的影响

在 SSD 层级卸载模型权重的设计中，**卸载粒度**、**单个 expert 粒度**和**SSD 读取粒度**三者之间存在深刻的不匹配关系，这是实现高效 SSD 卸载的核心技术难题。

### 5.1 三种粒度的定义

#### 卸载粒度（Offloading Granularity）

指模型权重从 DRAM 向 SSD 卸载时的最小可操作单元。在 llama.cpp 当前架构中：

- **最细粒度 = 整个 ggml_tensor**。每个 tensor（如 `blk.0.ffn_gate_exps`）是 `ggml_backend_buffer` 中的一次独立分配。
- **不支持子张量级卸载**。无法将 `ffn_gate_exps[:, :, expert_i]`（单个专家的切片）独立卸载到 SSD，而保留其他专家在 DRAM。
- **卸载决策在加载时一次完成**。`llama_model_loader::create_tensor()` 根据 buft（backend buffer type）规则为整个 tensor 分配目标 buffer，要么全在 DRAM，要么全在 GPU VRAM，之后不再改变。

```cpp
// llama-model-loader.cpp:1552 — 卸载决策位于 tensor 创建时
// 整个 ffn_gate_exps [n_embd, n_ff, n_expert] 作为一个单元
ggml_backend_tensor_alloc(buf_mmap, cur, data);  // 整个 tensor 分配
// 或
ggml_backend_tensor_set(cur, data, 0, n_size);    // 整个 tensor 拷贝
```

#### 单个 Expert 粒度（Single Expert Granularity）

指 MoE 模型中一个独立专家对应的权重数据量。在 Qwen3.5MoE 的 tensor 存储布局中：

- **所有专家的权重合并存储在同一个 tensor 中**。`ffn_gate_exps` 形状为 `[n_embd, n_ff, n_expert]`，在 GGUF 文件中是一个连续的数据块。
- **单个 expert 是该 tensor 在最后一个维度上的一个切片**：`ffn_gate_exps[:, :, expert_i]`。
- **计算时通过 ggml_mul_mat_id 索引选取**，但索引操作是在计算图执行阶段，不改变存储布局。

```cpp
// 专家权重 tensor 形状
// ffn_gate_exps: [n_embd, n_ff, n_expert]    ← 所有专家合并存储
//                                          ↑ expert 索引在最后一个维度
//
// 计算时通过索引选择（ggml_mul_mat_id 操作）
// 参数 ids = selected_experts [n_expert_used, n_tokens]
up = build_lora_mm_id(up_exps, cur, selected_experts);
// 物理上仍需整个 up_exps tensor 在内存中
```

单个 expert 的数据量估算（Qwen3.5-122B，Q4_K_M）：

```
假设: n_embd = 7168, n_ff = 20480, n_expert = 40, Q4_K_M ~0.5 bytes/elem

单个 expert 占用的连续数据量：
  gate_exps[:, :, expert_i] = n_ff × n_embd × 0.5 B
                             = 20480 × 7168 × 0.5
                             = 73.4 MB/层/专家/单矩阵

每层三个专家矩阵合计（gate + up + down）：
  3 × 73.4 MB = 220 MB/层/专家

若每层选 K=4 个专家，活跃权重：
  4 × 220 MB = 880 MB/层（活跃工作集）
```

#### SSD 读取粒度（SSD Read Granularity）

指从 SSD 读取模型权重数据时，一次 I/O 操作的最小和最优单元。涉及多个层次：

```
┌─────────────────────────────────────────────────────┐
│  OS mmap 缺页粒度                                      │
│  4KB (x86 标准页面) / 64KB (ARM) / 2MB (Huge Pages)   │
├─────────────────────────────────────────────────────┤
│  NVMe SSD 最优 I/O 单元                                │
│  通常 4KB ~ 128KB 对齐块                              │
│  （取决于 SSD 内部的 NAND 页面大小和控制器）              │
├─────────────────────────────────────────────────────┤
│  文件系统块大小                                        │
│  通常 4KB (ext4/NTFS)                                 │
├─────────────────────────────────────────────────────┤
│  GGUF 对齐要求                                        │
│  GGUF_DEFAULT_ALIGNMENT = 32 字节                     │
│  每个 tensor 在 GGUF 中的 offset 必须是 alignment 的倍数 │
├─────────────────────────────────────────────────────┤
│  单个 expert 连续数据                                   │
│  ~73 MB（一次跨越 ~18600 个 4KB 页面）                 │
└─────────────────────────────────────────────────────┘
```

关键代码路径：

```cpp
// llama-mmap.cpp:478 — unmap_fragment 的页面对齐
static void align_range(size_t * first, size_t * last, size_t page_size) {
    size_t offset_in_page = *first & (page_size - 1);
    size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
    *first += offset_to_page;
    *last = *last & ~(page_size - 1);
    if (*last <= *first) { *last = *first; }
}

// 页面大小为 sysconf(_SC_PAGESIZE) = 4096 (4KB) on x86
```

### 5.2 三种粒度的不匹配关系

三者之间存在**三个层次的不匹配**：

#### 不匹配 1：卸载粒度 vs Expert 粒度（粗 vs 细）

```
卸载粒度（tensor 级）：
  [================================================================]
  ffn_gate_exps: [n_embd, n_ff, n_expert=40]  →  约 2.9 GB/层/单矩阵

Expert 粒度（需要的数据）：
  [==expert_0==] [==expert_1==] ... [==expert_39==]
  每个 ~73 MB       每个 ~73 MB       每个 ~73 MB

不匹配：
  卸载粒度太大（2.9 GB），无法只卸载不需要的 36 个专家（2.6 GB）
  而保留需要的 4 个专家（0.3 GB）在 DRAM
```

后果：当前架构下，要么**整个 tensor 在 DRAM**（浪费 DRAM），要么**整个 tensor 在 SSD**（每次访问都要读大量数据）。无法实现"只保留活跃专家在 DRAM，不活跃专家在 SSD"的理想状态。

#### 不匹配 2：Expert 粒度 vs SSD 读取粒度（大 vs 小）

```
单个 expert 连续数据量：~73 MB
SSD 单次最优读取：4KB ~ 128KB（NVMe 典型值）

73 MB / 4 KB ≈ 18,672 个页面

不匹配：
  读取一个 expert 需要触发约 18,672 次缺页中断！
  或等效的 ~73 MB 顺序读取
```

后果：
- 如果每次访问一个 expert 都从 SSD 读取 ~73 MB，当代 NVMe SSD 的带宽（~7 GB/s）意味着 ~10ms 的延迟——这对于解码的实时性是不可接受的。
- 必须使用**预取**（prefetch）机制，在需要前就批量加载整个 expert 的数据。

#### 不匹配 3：三者的三角矛盾

```
                卸载粒度（粗 — tensor 级）
                    ↑
                   / \
                  /   \
                 /     \
                /       \
               /         \
              /           \
             /             \
    Expert 粒度（中 — ~73 MB） ←──→ SSD 读取粒度（细 — 4KB~128KB）
               不匹配 2

三者的矛盾三角：
  - 卸载粒度太粗 → 无法精细选择要卸载的内容
  - Expert 粒度处于中间 → 太大无法高效从 SSD 随机读取
  - SSD 读取粒度太细 → 需要大量页面才能凑齐一个 expert
```

### 5.3 这种不匹配如何影响 SSD 卸载方案设计

#### 影响 1：必须重新设计专家权重在 GGUF 文件中的布局

当前布局下，所有专家在同一个 tensor 中，数据在文件中连续排列：

```
当前 GGUF 布局（连续存储）：
  blk.0.ffn_gate_exps: [expert_0 | expert_1 | ... | expert_39]  ← 一个连续块
  偏移:  0          73MB     146MB     ...    2920MB
  （每 73MB = 一个 expert 在最后一个维度上的切片大小）

理想布局（按 expert 对齐到页边界）：
  blk.0.ffn_gate_exps:
    expert_0: [4KB align | data | padding]  ← 每个 expert 对齐到页边界
    expert_1: [4KB align | data | padding]
    ...
    expert_39: ...
```

但问题是：当前 GGUF 的 tensor 写入要求 `GGML_ASSERT(ggml_is_contiguous(&info.t))`，即 tensor 在文件中必须是连续的。要打破连续性，需要：

- **修改 GGUF 格式**：允许 expert 级别的非连续存储，或
- **在加载时创建多个 view tensor**：每个 expert 一个 ggml_tensor，通过 `create_tensor_as_view` 使用偏移量指向主干 tensor 的不同位置

#### 影响 2：mmap 必须先预读整个 expert，不能按页按需读取

因为单个 expert 的 ~73MB 数据在计算时必须**全部可用**（`ggml_mul_mat_id` 将整个 `[n_ff, n_embd]` 切片参与矩阵乘法），如果使用 mmap 的 demand paging（缺页按需加载），会导致第一次访问该 expert 时产生大量连续的 page fault，解码速度急剧下降。

```cpp
// 当前行为：mmap 后第一次访问 expert_i 的权重时
// 触发 ~18,672 次缺页中断，每个 ~10μs → 总计 ~186ms 的延迟
cur->data = (uint8_t *)mapping->addr() + w.offs;
// w.offs 指向 blk.0.ffn_gate_exps 的起始
// 访问 expert_i = mapping->addr() + w.offs + i * 73MB
// → 该地址所在的页面（共 ~18,672 页）均未加载
```

解决方案必须是**预取整个 expert**：

```c
// 预取方案 1：posix_madvise(WILLNEED) 批量预读
void * expert_addr = (uint8_t *)mapping->addr() + w.offs + i * expert_size;
posix_madvise(expert_addr, expert_size, POSIX_MADV_WILLNEED);
// → 内核一次发起 ~73MB 的异步预读（通常是 128KB 批量的 readahead）

// 预取方案 2：显式 read 绕过 mmap
// 通过 aio / io_uring 发起异步 73MB 读取到 pin-memory buffer
// 后续从 pin-memory 快速拷贝到 GPU
```

#### 影响 3：需要引入"中间页缓存层"来桥接粒度差异

```
SSD (NVMe)        中间缓存 (DRAM)        GPU/PU (计算)
──────────         ──────────────        ─────────────
expert_0 (73MB) →  DRAM slot 0 (73MB) →  GPU compute
expert_3 (73MB) →  DRAM slot 1 (73MB) →  GPU compute
expert_7 (73MB) →  ...
                   ↑
              调度器在此做 granularity 转换：
              以 expert 为单位发起 SSD → DRAM 传输
              以 expert 为单位管理 DRAM → SSD 逐出
```

这个中间缓存层需要解决的核心问题：

| 问题 | 影响 |
|------|------|
| 缓存容量 | DRAM 中能同时放几个 expert？（~64GB 总 DRAM，减去系统和 KV cache，约剩 30-40GB，约可缓存 140-180 个 expert 切片——但注意每层有多层且还有非 expert 权重） |
| 缓存替换策略 | 用 LRU 还是 MoE-aware 的预取？（路由概率提供了专家热度的直接预测） |
| 预取时机 | 何时发起预取？（当前层 gate_inp 计算出路由概率后，立即预取下一层的热门专家） |
| 逐出时机 | 何时释放？（该层计算完成后，其专家权重可立即释放——MoE 按层访问的特性决定了专家权重有清晰的"使用后即释放"边界） |

#### 影响 4：Windows 下需要更大的设计改动

因为 `unmap_fragment` 在 Windows 上是空实现，mmap 页面一旦读入无法部分释放。在 Windows 上的替代方案有两种：

```cpp
// 方案 A：分片 mmap（每个 expert 独立映射）
// 每个 expert 使用独立的 CreateFileMapping + MapViewOfFile
// 释放时：UnmapViewOfFile(addr_of_expert_i)

// 方案 B：VirtualFree(MEM_DECOMMIT)
// 保留地址空间，但释放物理页面
// 由于 llama_mmap 用的是 CreateFileMapping，不能直接 VirtualFree
// 需要改用 VirtualAlloc + ReadFile 的模式
```

### 5.4 推荐的粒度对齐方案

综合以上分析，要实现高效的 SSD 卸载，必须对三个粒度进行**对齐**：

| 需要实现的对齐 | 方法 | 优先级 |
|----------------|------|--------|
| 卸载粒度 → Expert 粒度 | 将 ffn_gate_exps / ffn_up_exps / ffn_down_exps 拆分为 n_expert 个独立的 tensor（或 view tensor），每个 expert 可独立卸载 | **P0：必须做** |
| Expert 粒度 → 页边界对齐 | 修改 GGUF 格式或加载逻辑，确保每个 expert 的起始/结束地址与 OS 页边界（4KB）对齐，避免读取一个 expert 时"捎带"加载相邻 expert 的数据 | **P1：强烈建议** |
| SSD 读取粒度 → Expert 粒度 | 使用预读（prefetch）而非 demand paging，在路由决策完成后立即发起每个 expert 的整块顺序读取，充分利用 SSD 的顺序带宽 | **P0：必须做** |
| 中间缓存管理 | 在 DRAM 中实现 expert 级别的 LRU/MoE-aware 缓存，维护活跃工作集在 DRAM 容量内 | **P1：必须做** |

**核心结论**：三种粒度的不匹配是 SSD 卸载方案设计的根本矛盾。解决方向不是让三者趋同（这在物理上不可能），而是在中间缓存层做 granularity bridging——以 **Expert 粒度**作为统一的调度单元，将上层的粗粒度 tensor 拆解为多个细粒度的 expert 块，将下层的细粒度页面聚合成整块 expert 的预读单元。这个中间层是实现"64GB DRAM 跑 70GB 模型"的关键工程挑战。
