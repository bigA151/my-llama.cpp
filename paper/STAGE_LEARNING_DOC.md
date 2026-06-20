# 学习文档：llama.cpp 架构、ggml/ggml-vulkan 与 Qwen3.5MoE 计算图

> 现阶段要求：理解 llama.cpp 整体 code 架构，从上层 llama 到下层 ggml 和 ggml-vulkan；理解 MoE 模型，重点以 Qwen3.5MoE 为例掌握计算图构建。

## 1. 推荐阅读顺序

建议按“先看推理路径，再看模型结构，再看 backend”的顺序：

1. `include/llama.h`：公共 API，理解 `llama_decode`、model/context/batch 的概念。
2. `src/llama.cpp`、`src/llama-context.cpp`、`src/llama-model.cpp`：理解模型加载、上下文、KV/recurrent memory。
3. `src/llama-arch.cpp`、`src/llama-arch.h`：理解架构枚举、GGUF key、tensor name 映射。
4. `src/models/qwen35moe.cpp`：重点，Qwen3.5MoE 的 hparams、tensor 加载、主计算图、MTP 图。
5. `src/llama-graph.cpp`：重点，通用 graph builder，尤其 `build_moe_ffn()`。
6. `ggml/src/ggml.c`：理解 ggml tensor/op，尤其 `ggml_mul_mat_id()`。
7. `ggml/src/ggml-backend.cpp`：理解 backend scheduler、split、tensor copy。
8. `ggml/src/ggml-vulkan/ggml-vulkan.cpp` 和 `vulkan-shaders/`：理解 Vulkan pipeline、topk-moe、mul_mat_id。

## 2. llama.cpp 分层架构

可以把 llama.cpp 看成 6 层：

```text
应用层
  examples/server/cli/simple 等调用 llama API

llama API 层
  include/llama.h
  llama_model, llama_context, llama_batch, llama_decode

模型与图构建层
  src/llama-model.cpp
  src/models/*.cpp
  src/llama-graph.cpp

ggml 计算图层
  ggml_tensor
  ggml_cgraph
  GGML_OP_*

ggml backend 调度层
  ggml-backend.cpp
  backend buffer, scheduler split, tensor copy

设备后端层
  ggml-cpu
  ggml-vulkan
  ggml-cuda/metal/opencl 等
```

一次推理的简化路径：

```text
llama_decode(ctx, batch)
  -> 准备 ubatch / memory / output ids
  -> model.build_arch_graph(params)
  -> ggml_build_forward_expand(gf, final_node)
  -> ggml_backend_sched_graph_compute_async(...)
  -> backend split 执行 graph nodes
  -> logits/embeddings 写回
```

## 3. GGUF、hparams 和 tensor 映射

`llama-arch.cpp` 负责把 GGUF 中的 key/tensor 名称映射到内部结构。MoE 相关 key 包括：

- `expert_count` -> `hparams.n_expert`
- `expert_used_count` -> `hparams.n_expert_used`
- `expert_shared_count` -> `hparams.n_expert_shared`
- `expert_feed_forward_length` -> `hparams.n_ff_exp`
- `expert_shared_feed_forward_length` -> `hparams.n_ff_shexp`
- `expert_weights_scale`
- `expert_gating_func`

MoE 相关 tensor 包括：

- `LLM_TENSOR_FFN_GATE_INP`
- `LLM_TENSOR_FFN_GATE_UP_EXPS`
- `LLM_TENSOR_FFN_DOWN_EXPS`
- `LLM_TENSOR_FFN_GATE_INP_SHEXP`
- `LLM_TENSOR_FFN_GATE_SHEXP`
- `LLM_TENSOR_FFN_UP_SHEXP`
- `LLM_TENSOR_FFN_DOWN_SHEXP`

`llama-arch.cpp` 还把 `FFN_*_EXPS` 标记为 `GGML_OP_MUL_MAT_ID` 相关 tensor，这决定后端可以识别它们是 MoE 专家矩阵。

## 4. Qwen3.5MoE 模型结构

`src/models/qwen35moe.cpp` 是当前阶段最重要的文件。

### 4.1 超参数加载

`load_arch_hparams()` 做几件事：

- 读取专家 FFN 维度：`n_ff_exp`、`n_ff_shexp`。
- 读取 gated delta net/linear attention 参数：`ssm_d_conv`、`ssm_d_inner`、`ssm_d_state`、`ssm_dt_rank`、`ssm_n_group`。
- 读取 `nextn_predict_layers`，用于 MTP/NextN。
- 根据 `full_attn_interval` 标记 recurrent layers。
- 根据主干层数识别模型类型：40 -> 35B-A3B，48 -> 122B-A10B，60 -> 397B-A17B。

这说明 Qwen3.5MoE 不是纯 attention 模型，而是 hybrid：部分层 full attention，部分层 recurrent/linear attention。

### 4.2 Tensor 加载

每个主干层加载：

```text
attention 或 recurrent tensors
attn_norm
attn_post_norm

routed experts:
  ffn_gate_inp
  ffn_gate_up_exps
  ffn_down_exps

shared experts:
  ffn_gate_inp_shexp
  ffn_gate_shexp
  ffn_up_shexp
  ffn_down_shexp
```

Qwen3.5MoE 使用 fused gate/up expert tensor：

```text
ffn_gate_up_exps: [n_embd, 2 * n_ff_exp, n_expert] 逻辑含义
```

在 ggml 的 `mul_mat_id` 视角下，它按 `[cols, rows, n_expert]` 存储；实际代码中会通过 `build_lora_mm_id()` 处理转置语义。

### 4.3 主计算图

主图结构：

```text
token embedding
  -> for each main layer:
       attn_norm
       full attention 或 gated delta net
       attention residual
       attn_post_norm
       MoE FFN
       FFN residual
  -> output_norm
  -> LM head
```

关键点：

- `hparams.is_recurrent(il)` 决定这一层走 full attention 还是 linear attention。
- MTP/NextN 额外 block 被加载，但主图不执行它；MTP 图单独处理。
- MoE FFN 在 attention residual 之后、FFN residual 之前。

## 5. Qwen3.5MoE 的 FFN 计算图

### 5.1 build_layer_ffn()

`build_layer_ffn(cur, il)` 做两部分：

```text
routed_moe = build_moe_ffn(...)

if shared expert exists:
  shared = build_ffn(...)
  shared_gate = sigmoid(build_lora_mm(ffn_gate_inp_shexp, cur))
  out = routed_moe + shared * shared_gate
else:
  out = routed_moe
```

Qwen3.5MoE 的 routed expert 调用重点参数：

- `gate_inp = layer.ffn_gate_inp`
- `down_exps = layer.ffn_down_exps`
- `gate_up_exps = layer.ffn_gate_up_exps`
- `n_expert = hparams.n_expert`
- `n_expert_used = hparams.n_expert_used`
- `gating_op = SOFTMAX`
- `type_op = SILU`

### 5.2 build_moe_ffn() 流程

以输入 `cur = [n_embd, n_tokens]` 为例：

```text
1. router logits:
   logits = gate_inp^T @ cur
   shape: [n_expert, n_tokens]

2. router probs:
   probs = softmax(logits)
   shape: [n_expert, n_tokens]

3. top-k:
   selected_experts = argsort_top_k(probs, n_expert_used)
   shape: [n_expert_used, n_tokens]

4. weights:
   weights = get_rows(probs, selected_experts)
   shape: [1, n_expert_used, n_tokens]

5. reshape hidden:
   cur = reshape_3d(cur, n_embd, 1, n_tokens)

6. fused gate/up expert matmul:
   gate_up = mul_mat_id(gate_up_exps, cur, selected_experts)
   shape: [2 * n_ff_exp, n_expert_used, n_tokens]

7. split + activation:
   gate = first half
   up   = second half
   hidden = silu(gate) * up

8. down expert matmul:
   experts = mul_mat_id(down_exps, hidden, selected_experts)
   shape: [n_embd, n_expert_used, n_tokens]

9. weighted sum:
   experts = experts * weights
   out = sum over n_expert_used
   shape: [n_embd, n_tokens]
```

### 5.3 为什么 `MUL_MAT_ID` 是项目核心

普通 dense FFN 每层只访问固定的 up/gate/down 矩阵；MoE FFN 每个 token 只访问 Top-K 专家矩阵。

`MUL_MAT_ID` 把“专家选择”直接编码进矩阵乘：

```text
as[:,:,expert_id] @ b[:, expert_rank, token]
```

因此内存优化要围绕两件事做：

- `selected_experts` 产生后，如何保证对应 expert 权重 ready。
- 未被选中的 expert 如何不占或少占 DRAM。

### 5.4 主计算图、FFN 计算图和 ggml 计算图的关系

这三个“计算图”不是并列关系，而是从模型语义到底层 IR 的三层表达：

```text
Qwen3.5MoE 主计算图
  整个 decoder forward 的模型级骨架
  代码位置：src/models/qwen35moe.cpp::graph::graph()
  语义：embedding -> N 层 block -> output_norm -> LM head

    其中每一层 block 内部包含：
      attention / recurrent 子图
      residual add
      attn_post_norm
      Qwen3.5MoE FFN 计算图
      residual add

Qwen3.5MoE FFN 计算图
  主计算图中每一层 MoE FFN 的子图
  代码位置：build_layer_ffn() -> build_moe_ffn()
  语义：router -> softmax/top-k -> routed experts -> shared expert -> add

ggml 计算图
  上面所有模型级节点最终落到的底层 tensor/op DAG
  代码位置：ggml_tensor / ggml_cgraph / ggml_build_forward_expand()
  语义：MUL_MAT、MUL_MAT_ID、SOFT_MAX、ARGSORT、GET_ROWS、ADD 等 op 依赖关系
```

换句话说：

- **主计算图**回答“Qwen3.5MoE 一次 forward 有哪些层，层与层如何串起来”。
- **FFN 计算图**回答“某一层的 MoE FFN 内部如何选专家、执行专家、合并输出”。
- **ggml 计算图**回答“这些模型级操作最终变成哪些 ggml tensor/op，backend scheduler 要执行哪些节点”。

可以把它理解为：

```text
模型结构视角:
  for layer in layers:
    attention_or_recurrent()
    moe_ffn()

MoE 子图视角:
  logits = router(hidden)
  ids, weights = top_k(softmax(logits))
  expert_out = mul_mat_id(experts, hidden, ids)
  out = weighted_sum(expert_out, weights) + shared_expert(hidden)

ggml IR 视角:
  ggml_mul_mat / ggml_soft_max / ggml_argsort_top_k /
  ggml_get_rows / ggml_mul_mat_id / ggml_mul / ggml_add
```

这里最容易混淆的一点是：`qwen35moe.cpp` 中写的是“模型级构图代码”，但它并不直接执行矩阵乘。它调用 `build_norm()`、`build_layer_attn()`、`build_moe_ffn()` 等 builder，这些 builder 会不断创建 `ggml_tensor`，并把每个 tensor 的 `op` 和 `src[]` 依赖填好。最后 `ggml_build_forward_expand(gf, cur)` 从输出节点反向收集依赖，形成真正交给 backend 的 `ggml_cgraph`。

对专家卸载来说，这个层级关系很重要：

- 在 **主计算图层级**，我们知道当前执行到哪一层，能做 layer-aware 预算和预取。
- 在 **FFN 子图层级**，`selected_experts` 已经产生，是专家 cache/on-demand/prefetch 的关键信号。
- 在 **ggml 计算图层级**，`GGML_OP_MUL_MAT_ID` 真正消费 expert 权重，backend scheduler 可以根据 ids 做 used expert copy 或绑定 expert cache window。

## 6. ggml 计算图基础

ggml 的几个概念：

- `ggml_tensor`：张量，包含 shape `ne[]`、stride `nb[]`、op、src、buffer。
- `ggml_cgraph`：计算图，节点是 tensor/op。
- `ggml_build_forward_expand()`：从输出 tensor 反向扩展依赖，加入 graph。
- `GGML_OP_*`：op 类型，如 `MUL_MAT`、`MUL_MAT_ID`、`SOFT_MAX`、`ARGSORT`。

`ggml_mul_mat_id()` 的形状定义：

```text
as  -> [cols, rows, n_expert]
b   -> [cols, n_expert_used, n_tokens]
ids -> [n_expert_used, n_tokens]
c   -> [rows, n_expert_used, n_tokens]
```

这和 Qwen3.5MoE 的 routed experts 非常吻合：每个 expert 是一个矩阵，第三维是 expert id。

## 7. ggml backend scheduler

backend scheduler 负责把图切成若干 split，让不同 backend 执行。

核心概念：

- tensor 可能在 CPU、Vulkan、CUDA、Metal 等 backend buffer 中。
- split 执行前要把输入 tensor copy 到目标 backend。
- 对普通权重，copy 往往是整个 tensor。
- 对 `MUL_MAT_ID` 权重，当前代码已有优化：只复制 ids 中实际用到的 expert slice。

这个优化是项目的重要起点。它说明只要能让 expert slice 的数据来源从“常驻 host tensor”变成“expert cache/mmap window”，后端侧不一定需要马上大改。

## 8. ggml-vulkan MoE 路径

Vulkan 后端主要文件：

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl`

### 8.1 fused topk-moe

`topk_moe.comp` 将 softmax/sigmoid、Top-K、权重输出融合成一个 shader。它输出：

- selected expert ids。
- selected expert weights。

这对性能很重要，因为 router 的 expert 数通常较多，但每 token 最后只保留少量 Top-K。

### 8.2 Vulkan MUL_MAT_ID

Vulkan 的 matmul shader 通过 `MUL_MAT_ID` 宏启用 expert id 路径：

- 从 ids buffer 读取 `expert_id`。
- 根据 `expert_id` 计算权重 tensor 的 expert offset。
- 对输入 token/expert rank 执行矩阵乘。

对 Q4_K_M 这类量化类型，Vulkan 会走 dequant + matmul/matvec 相关 pipeline。文档阶段需要知道的是：shader 消费的是已经绑定到 Vulkan buffer 的权重，不负责从 SSD 加载专家。

## 9. MoE 专家卸载应插入在哪里

不建议第一步改 shader。推荐插入点按优先级：

1. `llama-model-loader.cpp` / model tensor allocation：决定 routed expert 是否全量常驻。
2. `ggml-backend.cpp` scheduler：在 `MUL_MAT_ID` split copy 前确认 used experts。
3. `llama-graph.cpp` `build_moe_ffn()`：暴露/记录 `selected_experts`，触发 predictor。
4. `ggml-vulkan.cpp`：优化 expert cache window 到 Vulkan buffer 的绑定与复制。

原因：

- Router 计算完成后才知道本层专家。
- Host scheduler 能做 mmap/SSD I/O 和 cache 状态机。
- Backend shader 保持现有 `MUL_MAT_ID` 语义，可减少风险。

## 10. 当前阶段学习任务清单

### Week 1：llama.cpp 主路径

- 阅读 `include/llama.h` 的 API。
- 跑通 `llama_decode` 到 graph compute 的调用链。
- 理解 `llama_model`、`llama_context`、`llama_batch`。
- 输出一张 decode 调用链图。

### Week 2：Qwen3.5MoE 计算图

- 精读 `src/models/qwen35moe.cpp`。
- 手画每层 full attention/recurrent/MoE/residual 路径。
- 精读 `build_layer_ffn()` 和 `build_moe_ffn()`。
- 用形状表追踪 `[n_embd, n_tokens] -> [n_embd, n_tokens]`。

### Week 3：ggml 与 backend

- 阅读 `ggml_mul_mat_id()`。
- 阅读 `ggml_backend_sched_compute_splits()` 中 MoE used expert copy 优化。
- 理解 tensor buffer、backend buffer、copy tensor。
- 记录哪些 tensor 是 weights、哪些是 graph temporary。

### Week 4：Vulkan 后端

- 阅读 `topk_moe.comp`。
- 阅读 `mul_mm.comp` / `mul_mmq.comp` 的 `MUL_MAT_ID` 分支。
- 在 `ggml-vulkan.cpp` 中定位 pipeline 创建和 op dispatch。
- 理解 Q4_K_M 量化 matmul pipeline。

### Week 5：专家缓存原型准备

- 给 `selected_experts` 加 tracing。
- 统计 2K/8K/32K 每层 unique experts。
- 模拟 LRU/LFU/priority cache。
- 根据结果决定 cache 粒度。

## 11. 必须掌握的关键问题

学习结束时，应能回答这些问题：

- Qwen3.5MoE 哪些层是 recurrent，哪些层是 full attention？
- `nextn_predict_layers` 为什么不在主图执行？
- routed expert 和 shared expert 的 tensor 分别是什么？
- `ffn_gate_up_exps` 为什么要 split 成 gate/up？
- `selected_experts` 的 shape 是什么？
- `MUL_MAT_ID` 的 `as/b/ids/output` shape 各是什么？
- 当前 backend scheduler 如何只复制 used experts？
- 为什么这只能降低 backend copy，不能直接降低 DRAM？
- Vulkan shader 在哪里读取 expert id？
- 专家卸载为什么应该先改 loader/cache/scheduler，而不是先改 shader？

## 12. 术语表

| 术语 | 含义 |
|---|---|
| MoE | Mixture-of-Experts，混合专家模型 |
| Router/Gate | 为每个 token 选择专家的门控网络 |
| Top-K | 每个 token 选出的 K 个专家 |
| Routed Expert | 通过 router 选择的专家 |
| Shared Expert | 每个 token 都执行的共享专家 |
| `MUL_MAT_ID` | 按 expert id 选择矩阵的 MoE 矩阵乘 |
| Prefill | 提示词解析阶段，一次处理多个 token |
| Decode | 自回归生成阶段，通常每步处理一个或少量 token |
| KV cache | full attention 层保存 K/V 的缓存 |
| Recurrent state | linear attention/recurrent 层保存的状态 |
| Expert cache | 热门专家权重驻留缓存 |
| On-demand | 发现需要某专家后才加载 |
| Prefetch | 预测未来需要某专家并提前加载 |
| MTP/NextN | 多 token prediction 额外预测头 |
