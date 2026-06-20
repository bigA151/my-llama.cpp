# 学习文档：llama.cpp 架构与 MoE 模型（Qwen3.5）计算图

> 目标：理解如何在 AMD 395（64GB 统一内存）平台上，通过专家卸载和预加载技术，高效运行 Qwen3.5-122B-A10B-Q4_K_M

---

## 第一部分：核心概念与原理

### 1.1 什么是 MoE（Mixture-of-Experts）

MoE 是一种模型架构设计，将传统 FFN（前馈网络）层替换为多个并行的专家子网络，
每个 token 只激活其中一小部分。

关键参数：
- n_expert（总专家数，如 Qwen3.5 有 80 个专家）
- n_expert_used（每个 token 激活的专家数，如 8）
- n_expert_shared（共享专家，始终激活）

Qwen3.5-122B-A10B 具体参数：
- 总参数：~122B
- 每 token 活跃参数：~10B（所以叫 A10B）
- Q4_K_M 量化后：~68.6 GB（122B x 4.5 bpw）
- 为什么跑不了：AMD 395 只有 64GB 统一内存，必须做专家卸载

### 1.2 专家路由机制

每个 MoE 层有一个路由门控矩阵（ffn_gate_inp，形状 [n_embd, n_expert]）。

> 注：以下公式是 llama2.c 风格的概念原型，用于理解 MoE 路由的核心数学原理。
> 实际的 llama.cpp 实现（`build_moe_ffn`）使用 `build_lora_mm`（转置 matmul）
> 且各步骤的变量命名和顺序有差异。

```
# 路由计算
logits = gate_inp^T @ hidden_state     -> [n_expert, n_tokens]
probs  = softmax(logits)               -> [n_expert, n_tokens]
selected_experts = top_k(probs, k)   -> [n_expert_used, n_tokens]
weights = gather(probs, selected_experts)-> [1, n_expert_used, n_tokens]

# 选中的 expert 执行 FFN
up   = mul_mat_id(ffn_up_exps[id], hidden_state)
gate = mul_mat_id(ffn_gate_exps[id], hidden_state)
cur  = silu(gate) * up
out  = mul_mat_id(ffn_down_exps[id], cur)
result = sum_i(weights[i] * out[i])
```

### 1.3 共享专家（Shared Experts）

Qwen3.5 除了多个路由专家外，还有一个共享专家（对每个 token 始终激活）：
- ffn_gate_inp_shexp：共享专家的门控（输出 sigmoid 值控制贡献度）
- ffn_up_shexp / ffn_gate_shexp / ffn_down_shexp：共享专家权重
- 最终输出 = MoE 输出 + sigmoid(shared_gate) x shared_expert_output

### 1.4 统一内存架构（AMD 395）

AMD 395 的特殊之处：
- CPU 和 GPU 共享同一物理 64GB DRAM
- 没有传统 GPU 的 PCIe 数据传输瓶颈
- "卸载"在这里的含义：控制 working set 大小，让 GPU 只访问当前需要的专家
- 非活跃专家可以 memory-map 从 SSD 按需加载
- 专家预取从 PCIe 延迟问题变成了 SSD I/O 调度问题

---

## 第二部分：llama.cpp 整体代码架构

### 2.1 架构分层

```
应用层 (server/main)
API层 (llama.h: llama_decode, llama_model_load)
模型层 (src/models/qwen35moe.cpp: build_arch_graph)
计算图层 (llama-graph.cpp: build_moe_ffn)
GGML张量层 (ggml.c: ggml_mul_mat, ggml_mul_mat_id)
后端抽象层 (ggml-backend.cpp: ggml_backend_graph_compute)
GPU后端 (ggml-vulkan/ 或 ggml-cuda/ 或 ggml-cpu/)
```

### 2.2 核心文件一览

| 文件 | 角色 |
|---|---|
| include/llama.h | 公共 C API，暴露 llama_decode 等 |
| src/llama-impl.cpp | 核心推理循环实现 |
| src/llama-arch.cpp | 所有架构注册与 tensor 名称映射 |
| src/llama-graph.cpp | 通用 MoE 计算图（build_moe_ffn） |
| src/models/qwen35moe.cpp | Qwen3.5MoE 模型定义与 graph |
| src/llama-hparams.h | 超参数定义（n_expert 等） |
| src/llama-model-loader.cpp | 模型权重加载与分配 |
| src/llama-context.cpp | 上下文创建与 KV cache |
| ggml/src/ggml.c | 张量计算核心 |
| ggml/src/ggml-vulkan/ | Vulkan GPU 后端 |

### 2.3 推理流程（从 API 到底层）

```
llama_decode(ctx, batch)
  -> llama_decode_internal()  [src/llama-impl.cpp]
    -> model->build_arch_graph()  [一次构建，后续复用]
      -> for each layer:
        -> build_layer_ffn() -> build_moe_ffn()
          -> ggml_mul_mat / ggml_mul_mat_id
    -> ggml_backend_graph_compute(backend, gf)
      -> 遍历图节点，对每个 op:
         如果 op 在 GPU 上: vulkan_compute_forward()
         如果 op 在 CPU 上: ggml_compute_forward_mul_mat()
  -> 返回 logits
```

### 2.4 MoE 相关数据结构

**llama_layer**（每层包含的 tensor 指针）：

```cpp
struct llama_layer {
    // MoE 路由专家
    ggml_tensor * ffn_gate_inp;       // [n_embd, n_expert]       路由器权重（转置后为 [n_expert, n_embd]）
    ggml_tensor * ffn_up_exps;        // [n_embd, n_ff_exp, n_expert]  up 权重（Qwen3.5MoE 未使用）
    ggml_tensor * ffn_gate_exps;       // [n_embd, n_ff_exp, n_expert]  gate 权重（Qwen3.5MoE 为 nullptr）
    ggml_tensor * ffn_down_exps;      // [n_ff_exp, n_embd, n_expert] down 权重
    ggml_tensor * ffn_gate_up_exps;   // [n_ff_exp, n_embd, n_expert] 合并的 gate+up（Qwen3.5MoE 使用此路径）

    // 共享专家
    ggml_tensor * ffn_gate_inp_shexp;  // [n_embd]  共享门控（1D 向量）
    ggml_tensor * ffn_up_shexp;        // [n_embd, n_ff_shexp]
    ggml_tensor * ffn_gate_shexp;      // [n_embd, n_ff_shexp]
    ggml_tensor * ffn_down_shexp;      // [n_ff_shexp, n_embd]
};
```

> **注意**：Qwen3.5MoE 不使用分离的 `ffn_up_exps` 和 `ffn_gate_exps`，这两个字段为 nullptr。实际使用合并的 `ffn_gate_up_exps`（形状 `[n_ff_exp, n_embd, n_expert]`），其内部在 `build_moe_ffn` 中被 split 为 gate/up 两部分。

**llama_hparams**（关键超参字段）：

```cpp
struct llama_hparams {
    uint32_t n_expert = 0;        // 总专家数（示例值: 80）
    uint32_t n_expert_used = 0;   // 每 token 激活专家数（示例值: 8）
    uint32_t n_expert_shared = 0; // 共享专家数
    int64_t  n_ff_exp;            // 每个专家 FFN 中间维度
    float    expert_weights_scale;
    uint32_t expert_gating_func;  // SOFTMAX, SIGMOID 等
};
```

### 2.5 关键 GGML Op（MoE 专用）

**ggml_mul_mat_id** — 带索引选择的矩阵乘法，MoE 的核心 Op：

```
普通 mul_mat:    C = A x B               (一个矩阵乘法)
mul_mat_id:      C[id] = A[id] x B       (多个矩阵按索引选择)
  A 是 3D tensor [n_ff_exp, n_embd, n_expert]  <- 所有专家堆在一起（Qwen3.5MoE 的 ffn_gate_up_exps 形状）
  id 是 selected_experts [k, n_tokens]     <- 每个 token 选哪些专家
  结果 C 是 [n_ff, k, n_tokens]            <- 每 expert 各有一个输出
```

**ggml_argsort_top_k** — 选择概率最高的 K 个专家索引


---

## 第三部分：Qwen3.5MoE 计算图构建详解

### 3.1 架构注册链

```
llama-arch.cpp: 声明 LLM_ARCH_QWEN35MOE -> "qwen35moe"
models/models.h: struct llama_model_qwen35moe : public llama_model_base
models/qwen35moe.cpp: 实现三个核心方法:
  1. load_arch_hparams()  - 加载超参 (n_expert, n_ff_exp等)
  2. load_arch_tensors()  - 创建 tensor（包括专家 3D tensor）
  3. build_arch_graph()   - 构建计算图
     graph::graph() - 标准解码图
     graph_mtp::graph_mtp() - MTP 预测头
```

### 3.2 模型加载（load_arch_tensors）

从 gguf 文件读取后，为每层创建专家 tensor：

```
// 路由专家（核心）
layer.ffn_gate_inp  = create_tensor(
    LLM_TENSOR_FFN_GATE_INP,  [n_embd, n_expert])
layer.ffn_down_exps = create_tensor(
    LLM_TENSOR_FFN_DOWN_EXPS, [n_ff_exp, n_embd, n_expert])
layer.ffn_gate_up_exps = ... (合并 gate+up)

// 共享专家
layer.ffn_gate_inp_shexp = ...  [n_embd]
layer.ffn_gate_shexp     = ...  [n_embd, n_ff_shexp]
layer.ffn_up_shexp       = ...  [n_embd, n_ff_shexp]
layer.ffn_down_shexp     = ...  [n_ff_shexp, n_embd]
```

**关键理解**：所有专家的权重存放在同一个 3D tensor 中
（维度 `[n_ff_exp, n_embd, n_expert]`），第三维索引不同专家。
这意味着当前无法按单个 expert 粒度进行内存管理——
专家卸载需要修改这个数据结构为按 expert 分块存储。

### 3.3 计算图主循环（graph::graph 构造函数）

Qwen3.5MoE 是 hybrid 模型（部分 attention + 部分 linear attention/SSM）：

```python
inpL = build_inp_embd(model.tok_embd)  # 输入嵌入

for il in range(n_transformer_layers):
    # 1. 注意力前归一化
    cur = build_norm(inpL, layer.attn_norm, RMS)

    # 2. 注意力层（二选一）
    if hparams.is_recurrent(il):
        # gated delta net（线性注意力/SSM）
        cur = build_layer_attn_linear(inp->get_recr(), cur, il)
    else:
        # 完整注意力
        cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il)

    # 3. 残差连接
    cur = ggml_add(ctx0, cur, inpSA)
    ffn_residual = cur

    # 4. FFN 前归一化
    attn_post_norm = build_norm(cur, layer.attn_post_norm, RMS)

    # 5. MoE FFN 层（核心！）
    cur = build_layer_ffn(attn_post_norm, il)

    # 6. FFN 残差连接
    cur = ggml_add(ctx0, cur, ffn_residual)
    inpL = cur  # 下一层输入

# 最终输出
cur = build_norm(inpL, model.output_norm, RMS)
cur = build_lora_mm(model.output, cur)  # LM head
```

**Hybrid 模型关键**：Qwen3.5MoE 的某些层使用标准 attention，
某些层使用 gated delta net（线性注意力/SSM），由 hparams.is_recurrent() 决定。
这对 expert 卸载无直接影响，但影响内存分配（需要区分 KV cache 和 recurrent state）。

### 3.4 build_layer_ffn() — Qwen3.5MoE FFN 层包装

在 qwen35moe.cpp:498 定义：

```python
def build_layer_ffn(cur, il):
    # 路由专家部分（使用合并的 ffn_gate_up_exps）
    # 注意：ffn_up_exps 和 ffn_gate_exps 在 Qwen3.5MoE 中均为 nullptr
    moe_out = build_moe_ffn(
        cur,
        layer.ffn_gate_inp,        # [n_embd, n_expert] 路由器
        layer.ffn_up_exps,          # nullptr（未使用）
        layer.ffn_gate_exps,        # nullptr（未使用）
        layer.ffn_down_exps,        # [n_ff_exp, n_embd, n_expert]
        nullptr,                    # exp_probs_b（DeepSeek 用，Qwen3.5MoE 未用）
        n_expert, n_expert_used,
        LLM_FFN_SILU, true,
        hparams.expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
        nullptr,                     # probs_in
        layer.ffn_gate_up_exps,    # 合并的 gate+up（[n_ff_exp, n_embd, n_expert]）
        layer.ffn_up_exps_s,        # up 的 smooth quant 缩放（可为空）
        layer.ffn_gate_exps_s,     # gate 的 smooth quant 缩放（可为空）
        layer.ffn_down_exps_s)      # down 的 smooth quant 缩放（可为空）

    # 共享专家部分（如果存在）
    if layer.ffn_up_shexp:
        # build_ffn 是通用的 FFN 构建器，内部处理 swiglu
        ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, NULL,   layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, NULL, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, NULL,  layer.ffn_down_shexp_s,
            NULL)
        # 共享门控：ffn_gate_inp_shexp 是 1D 向量 [n_embd]
        # build_lora_mm 等于转置 matmul: (gate_inp_shexp)^T @ cur
        shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur)
        shared_gate = sigmoid(shared_gate)          # -> [n_tokens]
        cur = moe_out + ffn_shexp * shared_gate
    else:
        cur = moe_out
    return cur
```

> `build_lora_mm(A, B)` 在 llama.cpp 中等价于 `A^T @ B`（转置矩阵乘法），而非普通 `mul_mat`。
> LoRA adapter 的权重 A 通常是 `[out_dim, in_dim]`，`build_lora_mm(A, B)` 执行的是
> `A^T @ B`，与普通 `ggml_mul_mat`（`A @ B`）方向相反。这是 LoRA adapter 的标准做法，因此命名如此。

### 3.5 build_moe_ffn() 分步详解

在 llama-graph.cpp:1368 定义。以下按 Qwen3.5MoE 的实际路径（使用合并的 `ffn_gate_up_exps`）分步说明：

**Step 1: 路由计算**

```
# build_lora_mm = 转置 matmul，即 gate_inp^T @ cur
logits = build_lora_mm(gate_inp, cur)         -> [n_expert, n_tokens]
probs  = softmax(logits)                      -> [n_expert, n_tokens]

if n_expert_groups > 1:  # DeepSeek 式分组路由
    # 组织成组 -> 选 top-n 组 -> mask 掉其他组
    selection_probs = masked_probs

selected_experts = argsort_top_k(probs, n_used)  -> [n_used, n_tokens]
weights = get_rows(probs, selected_experts)      -> [1, n_used, n_tokens]
```

**Step 2: 合并 gate+up 的专家 FFN（Qwen3.5MoE 路径）**

```
# Qwen3.5MoE 使用 ffn_gate_up_exps（合并的 gate+up 权重）
# 形状为 [n_ff_exp, n_embd, n_expert]
# build_moe_ffn 内部会 split 为 gate/up 两部分

gate_up = mul_mat_id(gate_up_exps, cur, selected_experts)
  gate_up_exps = [n_ff_exp, n_embd, n_expert]
  -> gate_up = [n_ff_exp, n_used, n_tokens]

# 内部 split（不在 build_layer_ffn 中暴露，由 build_moe_ffn 处理）：
gate = view(gate_up[0:n_ff_exp, :, :])   # 前半 [n_ff_exp/2, n_used, n_tokens]
up   = view(gate_up[n_ff_exp/2:, :, :]) # 后半 [n_ff_exp/2, n_used, n_tokens]
```

> **其他模型（如 DeepSeek）的路径**：不使用 `ffn_gate_up_exps`，而是传入分离的
> `ffn_up_exps`（形状 `[n_embd, n_ff_exp, n_expert]`）和
> `ffn_gate_exps`（形状 `[n_embd, n_ff_exp, n_expert]`），分别通过各自的
> `mul_mat_id` 得到 up 和 gate。

**Step 3: 激活函数**

```
cur = swiglu(gate, up) = silu(gate) * up
  -> [n_ff_exp/2, n_used, n_tokens]
```

**Step 4: 下行投影**

```
# 注意：此处的 cur 是 Step 3 的输出（激活后），非原始隐藏状态
experts = mul_mat_id(down_exps, cur, selected_experts)
  down_exps = [n_ff_exp, n_embd, n_expert]
  -> [n_embd, n_used, n_tokens]
```

**Step 5: 加权求和**

```
experts = mul(experts, weights)  # 按权重缩放
moe_out = sum(experts[i] for i in 0..n_used)
  -> [n_embd, n_tokens]  # 回到原始形状
```

### 3.6 张量形状追踪示例

以 prefill 阶段（2048 tokens）为例，假设 n_ff_exp = 4096（即每个 expert 的 FFN 中间维度）：

```
输入: cur = [n_embd=2048, n_tokens=2048]

Step 1: build_lora_mm(gate_inp, cur) -> [n_expert=80, n_tokens=2048]
  probs = softmax(logits) -> [80, 2048]
  selected_experts = argsort_top_k(probs, n_expert_used=8) -> [8, 2048]
  weights = get_rows(probs, selected_experts) -> [1, 8, 2048]

Step 2: mul_mat_id(ffn_gate_up_exps, cur, selected_experts)
  # ffn_gate_up_exps = [n_ff_exp=4096, n_embd=2048, n_expert=80]
  # split 后得到 [2048, 8, 2048] -> 拆分为:
  gate = [n_ff_exp/2=2048, 8, 2048]
  up   = [n_ff_exp/2=2048, 8, 2048]

Step 3: silu(gate) * up -> [2048, 8, 2048]

Step 4: mul_mat_id(down_exps, cur, selected_experts)
  # 注意：此处的 cur 是 Step 3 的激活输出，非原始隐藏状态
  # down_exps = [n_ff_exp=2048, n_embd=2048, n_expert=80]
  # cur (激活后) = [n_ff_exp/2=2048, 8, 2048]
  experts = [2048, 8, 2048]

Step 5: mul(experts, weights) -> sum(experts * weights) -> [2048, 2048]  回到原始形状
```

关键观察：第 2 维从 n_tokens(2048) 变为 n_expert_used(8)，
意味着每个 token 的专家输出被堆在同一个 tensor 的不同切片中。
这要求所有被选中的 expert 权重必须在 GPU 上可用，
因此是 expert 卸载需要解决的核心问题。

### 3.7 共享专家在 Qwen3.5MoE 中的位置

参考代码（qwen35moe.cpp:519-548）：

```
if layer.ffn_up_shexp:
    # 共享专家 FFN（始终激活，无路由选择）
    ffn_shexp = build_ffn(cur,
        layer.ffn_up_shexp, NULL,
        layer.ffn_gate_shexp, NULL,
        layer.ffn_down_shexp, NULL,
        NULL)  # -> [n_embd, n_tokens]

    # 共享门控：ffn_gate_inp_shexp 是 1D 向量 [n_embd]
    # build_lora_mm 等于转置 matmul: (gate_inp_shexp)^T @ cur
    shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur)
    shared_gate = sigmoid(shared_gate)            # -> [n_tokens]

    # 合并
    cur = moe_out + ffn_shexp * shared_gate
```

> `build_lora_mm(A, B)` 等于 `A^T @ B`。这里 `ffn_gate_inp_shexp` 是形状 `[n_embd]` 的 1D 向量，
> `cur` 是形状 `[n_embd, n_tokens]` 的 2D 张量，转置后 `[1, n_embd]` 再与 cur 相乘，
> 结果是 `[1, n_tokens]` 再 squeeze 成 `[n_tokens]`。

共享专家的意义：保证每个 token 都能访问一组基础知识，
不受路由决策的限制。对卸载策略来说，共享专家（体积小）
始终驻留在 DRAM 中，路由专家按需加载。

### 3.8 Vulkan GPU 后端中的 MoE 执行

```
ggml/src/ggml-vulkan/ 目录结构：
ggml-vulkan.cpp          # 后端入口：设备管理、管道创建、计算图提交（单文件，约 17k 行）
ggml-vulkan.h             # 头文件
vulkan-shaders/          # GLSL compute shader 目录
    *.comp                 # 各操作的 shader 源码（如 mul_mat_vec.comp、soft_max.comp）
    *.glsl                 # 被 *.comp include 的共享 GLSL 源码（如 rope_funcs.glsl）
    ggml-vulkan-shaders.hpp # 预编译的 SPIR-V bytecode（所有 shader 编译后嵌入）
```

`ggml_mul_mat_id` 在 Vulkan 后端的执行流程：
1. GGML 图调度器遍历所有 op 节点
2. 遇到 `GGML_OP_MUL_MAT_ID`:
   - Vulkan 后端根据 tensor 类型和数据布局查找对应的 shader
   - 将 selected_experts 索引 tensor 上传到 GPU
   - Dispatch compute shader，从 3D tensor 中按 expert id 切片执行矩阵乘法
3. 结果写回输出 tensor

> 注：实际代码中并不存在独立的 `mul_mat_id.comp` shader。`mul_mat_id` 的逻辑通过 Vulkan 后端
> 中的 `ggml_vk_mul_mat_id` 函数实现，内部根据量化类型和 tile 大小分发到对应的
> `mul_mat_vec` 或 `matmul` shader。

**学习重点**：要实现专家卸载，需要在 GGML 层新增
GGML_OP_MUL_MAT_ID_OFFLOAD 或在 Vulkan shader 中
新增 lazy loading 逻辑，让 shader 直接处理按需加载。

---

## 第四部分：学习路线图

### Phase 1：理解现有代码（2-3 周）

| 优先级 | 学习内容 | 关键文件 |
|---|---|---|
| ★★★ | MoE 核心路由计算 | llama-graph.cpp:1368-1764 |
| ★★★ | Qwen3.5MoE 模型定义 | src/models/qwen35moe.cpp |
| ★★★ | 模型加载与 tensor 注册 | llama-model-loader.cpp |
| ★★☆ | 推理主循环 | llama-impl.cpp |
| ★★☆ | 超参数定义 | llama-hparams.h |
| ★★☆ | Vulkan 后端 shader 管线 | ggml-vulkan/ |
| ★☆☆ | GGML 核心 op 定义 | ggml.c, ggml-impl.h |
| ★☆☆ | 张量形状与数据布局 | ggml-common.h |

### Phase 2：理解论文技术（1-2 周）

| 论文 | 核心贡献 | 学习重点 |
|---|---|---|
| ExpertFlow | 路由预测+预取+token 调度 | Predict->Schedule->Prefetch 流水线 |
| Speculating Experts | Quasi-hidden state 预取 | 残差流预测下一层路由 |
| FlashMoE | ML-based cache replacement | 超越 LRU/LFU 的混合策略 |
| Fast MoE Inference | SRU 预测+专家复制 | Batch 级预测与负载均衡 |

### Phase 3：实现方案（长期）

按实现顺序：
1. **专家 tensor 独立分配** — 修改 llama-model-loader.cpp
   当前：专家权重是 3D tensor [n_ff_exp, n_embd, n_expert]
   改为：每个专家分配独立 2D tensor，或 mmap 按需加载

2. **专家缓存管理** — 新增 src/llama-expert-cache.cpp
   DRAM expert cache（固定大小，如 32GB）
   LRU + 频率信号的混合 eviction 策略
   SSD -> DRAM 异步预取流水线

3. **路由预测器** — 在 llama-graph.cpp 中新增
   预填充阶段：执行所有 gate_inp 收集完整路由矩阵
   解码阶段：quasi-hidden state 预测下一层
   Token scheduling：K-means 聚类合并相似路由

4. **GGML op 扩展** — 在 ggml.c + Vulkan 后端
   新增 GGML_OP_MUL_MAT_ID_OFFLOAD 支持 lazy loading
   修改计算图调度器支持异步加载

---

## 附录：关键术语表

| 术语 | 含义 |
|---|---|
| MoE | Mixture-of-Experts，混合专家模型 |
| Expert | 专家子网络，MoE 层的独立 FFN |
| Router | 路由门控，决定 token 激活哪些专家 |
| Top-K | 选择概率最高的 K 个专家 |
| Shared Expert | 对所有 token 始终激活的公共专家 |
| mul_mat_id | 带索引选择的矩阵乘法（MoE 核心 Op） |
| Hybrid | 混合 attention+recurrent 层（Qwen3.5） |
| Unified Memory | CPU+GPU 共享 DRAM（AMD 395） |
| Quasi-Hidden State | 残差流预测下一层路由 |
| Expert Offloading | 将非活跃专家卸载到低层存储 |
| KVCache | Key-Value 缓存，解码避免重复计算 |
| MTP | Multi-Token Prediction（Qwen3.5 多 token 预测头）|

