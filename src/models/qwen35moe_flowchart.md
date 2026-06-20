# Qwen3.5 MoE 计算图构建流程图

> 本文档详细展示 `src/models/qwen35moe.cpp` 中计算图构建的完整流程。

---

## 1. 整体入口：`build_arch_graph`

```
┌─────────────────────────────────┐
│  build_arch_graph(params)       │
└──────────────┬──────────────────┘
               │
               ▼
      ┌───────────────────┐
      │ params.gtype ==   │
      │ LLM_GRAPH_TYPE_   │ YES
      │ DECODER_MTP ?     │
      └───────┬───────────┘
              │ NO
              ▼ (标准模式)
      ┌───────────────┐
      │  构建 graph   │
      │  (主计算图)    │
      └───────┬───────┘
              │ (MTP 模式)
              ▼
      ┌───────────────┐
      │ 构建 graph_mtp│
      │ (MTP 计算图)   │
      └───────────────┘
```

---

## 2. 主计算图 `graph::graph` — 完整层循环

```
输入: 输入 token 序列
  │
  ▼
┌─────────────────────────┐
│ 1. 词嵌入层              │
│ inpL = build_inp_embd() │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. 构建混合输入          │
│ inp = build_inp_mem_    │
│      hybrid()           │
│ (同时支持循环 & 注意力)  │
└────────────┬────────────┘
             │
             ▼
     ┌───────────────────┐
     │  初始化 inp_pos    │
     │  inp_out_ids      │
     └────────┬──────────┘
              │
              ▼
    ╔═══════════════════╗
    ║ for il in [0, N): ║
    ║  (N = 主解码层数)   ║
    ╠═══════════════════╣
    ║ Layer Loop Start  ║
    ╚═══════╤═══════════╝
             │
             ▼
┌─────────────────────────────┐
│ inpSA = inpL                │
│ (保存用于残差连接)            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ 3. 注意力前归一化 (RMS)       │
│ cur = build_norm(inpL,      │
│      attn_norm)             │
└────────────┬────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ hparams.is_     │ YES ──→ build_layer_attn_linear()
    │ recurrent(il)?  │         (线性注意力 / SSM)
    └────┬────────────┘
         │ NO
         ▼
    build_layer_attn()        build_layer_attn_linear()
    (标准多头注意力)          (详见第3节)
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌──────────────────────┐
         │ 4. 残差连接            │
         │ cur = cur + inpSA     │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │ ffn_residual = cur   │
         │ (保存用于 FFN 残差)    │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │ 5. 注意力后归一化 (RMS)│
         │ cur = build_norm(cur, │
         │      attn_post_norm)  │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │ 6. MoE 前馈网络层      │
         │ cur = build_layer_ffn │
         │      (attn_post_norm) │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │ 7. FFN 残差连接        │
         │ cur = cur + ffn_residual  │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │ 8. 控制向量处理        │
         │ cur = build_cvec(cur) │
         └──────────┬────────────┘
                    ▼
                    │ inpL = cur
                    │ (作为下一层输入)
                    │
         ╔══════════════════════╗
         ║ Layer Loop End        ║
         ╚════════╤═════════════╝
                  │
                  ▼
┌───────────────────────────────────────┐
│ 9. 最终归一化                           │
│ cur = build_norm(cur, output_norm)    │
└────────────┬──────────────────────────┘
             │
             ▼
┌───────────────────────────────────────┐
│ 10. LM Head                           │
│ cur = build_lora_mm(output, cur)      │
│ res->t_logits = cur                   │
└───────────────────────────────────────┘
             │
             ▼
输出: t_logits (词表维度的 logits)
```

---

## 3. 全注意力层 `build_layer_attn`（标准多头注意力 + 门控）

```
输入: cur (归一化后的隐藏状态)
  │
  ▼
┌─────────────────────────────────────────┐
│ Q 投影 (融合 Q + Gate)                   │
│ Qcur_full = build_lora_mm(Wq, cur)     │
│ 形状: [n_embd, 2*n_embd] (Q 和 Gate)    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 分割 Q 和 Gate (从同一矩阵切分)           │
│ Qcur = view_3d(Qcur_full, n_embd_head,   │
│                n_head, n_tokens)        │
│ gate = view_3d(Qcur_full, ..., offset)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Q 归一化 (RMS Norm)                      │
│ Qcur = build_norm(Qcur, attn_q_norm)   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ K, V 投影                               │
│ Kcur = build_lora_mm(Wk, cur)          │
│ Vcur = build_lora_mm(Wv, cur)          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ K 归一化 (RMS Norm)                      │
│ Kcur = build_norm(Kcur, attn_k_norm)   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 对 Qcur, Kcur 应用 IMRoPE               │
│ Qcur = ggml_rope_multi(Qcur, inp_pos)  │
│ Kcur = ggml_rope_multi(Kcur, inp_pos)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 计算注意力缩放因子                        │
│ kq_scale = 1/sqrt(n_embd_head)         │
│ (或使用配置的 f_attention_scale)         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 执行注意力计算                            │
│ cur = build_attn(Qcur, Kcur, Vcur)     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Gate Sigmoid                            │
│ gate_sig = sigmoid(gate)               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 门控注意力输出                           │
│ cur = cur * gate_sig                    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 输出投影                                 │
│ cur = build_lora_mm(Wo, cur)            │
└────────────┬────────────────────────────┘
             │
             ▼
输出: cur (注意力输出)
```

---

## 4. 线性注意力层 `build_layer_attn_linear`（SSM / Gated Delta Net）

```
输入: cur (归一化后的隐藏状态)
  │
  ▼
┌─────────────────────────────────────────────┐
│ [阶段 1] 输入投影                            │
│ qkvz = build_qkvz(cur, il)                  │
│  - qkv_mixed: build_lora_mm(wqkv, cur)       │
│  - z: build_lora_mm(wqkv_gate, cur)         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 2] 计算 SSM 门控参数                    │
│                                             │
│ Beta 路径:                                  │
│   beta = build_lora_mm(ssm_beta, cur)       │
│   beta = sigmoid(beta)  → 初始化 SSM 状态门  │
│                                             │
│ Alpha 路径:                                 │
│   alpha = build_lora_mm(ssm_alpha, cur)     │
│   alpha_biased = alpha + ssm_dt             │
│   alpha_softplus = softplus(alpha_biased)  │
│                                             │
│ Gate = alpha_softplus * ssm_a           │
│ (ssm_a 为对角矩阵参数, softplus 确保正值)  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 3] 获取循环状态                         │
│ conv_states = mctx->get_r_l(il)            │
│ ssm_states  = mctx->get_s_l(il)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 4] SSM 卷积分支                        │
│                                             │
│ conv_input = build_conv_state(             │
│              conv_states, qkv_mixed)       │
│ conv_output = ggml_ssm_conv(conv_input,    │
│                              ssm_conv1d)  │
│ conv_output = silu(conv_output)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 5] 从卷积输出中提取 Q, K, V            │
│                                             │
│ conv_qkv_mix 中提取:                        │
│   q_conv [head_k_dim, n_k_heads, ...]     │
│   k_conv [head_k_dim, n_k_heads, ...]     │
│   v_conv [head_v_dim, n_v_heads, ...]     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 6] RMS + L2 归一化                   │
│ q_conv = ggml_l2_norm(q_conv, eps)        │
│ k_conv = ggml_l2_norm(k_conv, eps)        │
│ (必要时 repeat 以匹配 n_v_heads)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 7] 循环注意力计算 (核心 SSM)           │
│ output = build_recurrent_attn(             │
│           ssm_states, q_conv, k_conv,      │
│           v_conv, gate, beta, state)        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 8] 门控归一化                          │
│ z_2d = reshape(z, [head_v_dim, n_v_heads,  │
│                    n_seq_tokens, n_seqs])  │
│ attn_out_norm = build_norm_gated(           │
│                  output, ssm_norm, z_2d)    │
│   即: output = RMSNorm(output) * silu(z)   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [阶段 9] 输出投影                           │
│ cur = build_lora_mm(ssm_out, attn_out_norm)│
│ cur = reshape(cur, [n_embd, n_seq_tokens]) │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
输出: cur (线性注意力输出)
```

---

## 5. MoE 前馈网络层 `build_layer_ffn`（路由专家 + 共享专家）

```
输入: cur (注意力后归一化的隐藏状态)
  │
  ▼
┌─────────────────────────────────────────────┐
│ [路由专家分支] MoE FFN                       │
│                                             │
│ 1. 路由器: 计算每个 token 对各专家的亲和度    │
│    gate_scores = build_lora_mm(ffn_gate_inp│
│                            cur)             │
│                                             │
│ 2. Softmax 归一化亲和度                      │
│    gate_probs = softmax(gate_scores)        │
│                                             │
│ 3. Top-K 选择 (选择 n_expert_used 个专家)     │
│    topk_indices, topk_weights               │
│                                             │
│ 4. 对每个选中的专家计算:                       │
│    - up:   build_lora_mm(ffn_up_exps, cur)  │
│    - gate: build_lora_mm(ffn_gate_exps,cur) │
│    - down: build_lora_mm(ffn_down_exps,    │
│                     silu(up*gate))          │
│                                             │
│ 5. 加权求和: moe_out = Σ(weight_i * down_i) │
│                                             │
│ gate_type = LLM_FFN_SILU                   │
│ gating_func = SOFTMAX                       │
│ moe_out = build_moe_ffn(...)               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
    ┌────────────────────────────┐
    │ 模型有共享专家?             │ NO ──→ 直接输出 moe_out
    │ ffn_up_shexp != nullptr?   │
    └────────────┬──────────────┘
                 │ YES
                 ▼
┌─────────────────────────────────────────────┐
│ [共享专家分支]                               │
│                                             │
│ 1. 计算共享专家门控 (标量，每个 token 一个)    │
│    shared_gate = build_lora_mm(             │
│              ffn_gate_inp_shexp, cur)       │
│                                             │
│ 2. FFN 前向:                                │
│    shexp_up   = build_lora_mm(ffn_up_shexp,│
│                               cur)           │
│    shexp_gate = build_lora_mm(ffn_gate_shexp│
│                               cur)           │
│    ffn_shexp  = build_lora_mm(ffn_down_shexp│
│                               silu(shexp_up*│
│                               shexp_gate))  │
│                                             │
│ 3. 门控共享专家:                            │
│    shared_gate = sigmoid(shared_gate)       │
│    ffn_shexp = ffn_shexp * shared_gate      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ [合并] 最终 FFN 输出                         │
│ cur = moe_out + ffn_shexp                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
输出: cur (FFN 输出)
```

---

## 6. MTP 计算图 `graph_mtp`（Multi-Token Prediction）

```
输入: 主模型最后一层的隐藏状态 h_input, 当前 token ids
  │
  ▼
┌─────────────────────────────────────────────────┐
│ [阶段 1] 准备输入                                │
│                                                   │
│ - h_input: 来自主模型的隐藏状态 (形状: [n_embd,  │
│   n_tokens])，通过 mtp_h_input 输入节点           │
│ - tokens: 当前 token id，通过 tokens 输入节点     │
│ - tok_embd: 从嵌入表中查找 token 对应的嵌入向量    │
│   tok_embd = get_rows(embed_tokens, tokens)      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ [阶段 2] E + H 融合 (MTP 核心设计)                │
│                                                   │
│ 对 h 和 e 分别归一化:                             │
│   h_norm = build_norm(h_input, hnorm)           │
│   e_norm = build_norm(tok_embd, enorm)          │
│                                                   │
│ 拼接: concat = [e_norm; h_norm] (dim=0)          │
│   形状: [2*n_embd, n_tokens]                    │
│                                                   │
│ 投影回 embedding 维度:                           │
│   cur = build_lora_mm(eh_proj, concat)          │
│   形状: [n_embd, n_tokens]                      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
      ┌────────────────────────────┐
      │ inpSA = cur                │
      │ (保存用于注意力残差连接)     │
      └──────────┬─────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ [阶段 3] 注意力子层 (与主图全注意力层相同结构)    │
│                                                   │
│ 注意力前归一化:                                   │
│   cur = build_norm(cur, attn_norm)               │
│                                                   │
│ Q/K/V 投影 + Q/K 归一化 + IMRoPE:                │
│   (与 build_layer_attn 相同的处理流程)            │
│                                                   │
│ 注意力计算:                                      │
│   cur = build_attn(Qcur, Kcur, Vcur)             │
│                                                   │
│ 门控注意力:                                      │
│   cur = cur * sigmoid(gate)                     │
│                                                   │
│ 输出投影:                                        │
│   cur = build_lora_mm(Wo, cur)                  │
│                                                   │
│ 注意力残差:                                      │
│   cur = cur + inpSA                             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
      ┌────────────────────────────┐
      │ ffn_residual = cur         │
      │ (保存用于 FFN 残差连接)      │
      └──────────┬─────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ [阶段 4] MoE FFN 子层                            │
│                                                   │
│ 注意力后归一化:                                   │
│   cur = build_norm(cur, attn_post_norm)           │
│                                                   │
│ MoE FFN:                                         │
│   moe_out = build_moe_ffn(cur, ...)              │
│                                                   │
│ 共享专家 (gate 来源于同一 cur):                   │
│   shexp = build_ffn(cur, ...)                   │
│   shared_gate = build_lora_mm(ffn_gate_inp_shexp,│
│                      cur)                         │
│   shexp = shexp * sigmoid(shared_gate)           │
│                                                   │
│ FFN 残差:                                        │
│   cur = moe_out + shexp + ffn_residual          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ [阶段 5] 预测头 (LM Head)                         │
│                                                   │
│ 提取输出位置:                                     │
│   cur = get_rows(cur, inp_out_ids)               │
│                                                   │
│ 归一化 (使用 shared_head_norm 或 output_norm):    │
│   cur = build_norm(cur, head_norm_w)             │
│                                                   │
│ LM Head:                                         │
│   cur = build_lora_mm(head_w, cur)               │
│   形状: [n_vocab, n_tokens]                      │
│                                                   │
│ res->t_logits = cur                              │
│ res->t_h_pre_norm = cur (用于 AR 草稿循环播种)    │
└─────────────────────────────────────────────────┘
                   │
                   ▼
输出: t_logits (预测下一个 token 的 logits)
```

---

## 7. 层类型分布（每 4 层一个循环）

以下以 40 层主解码器 + 1 层 MTP 为例（Qwen2.5-35B-A3B）：

```
层索引:    0    1    2    3    4    5    6    7   ...  37   38   39   40(OTP)
          ─────────────────────────────────────────────────────  ────
注意力类型: FA   LA   LA   LA   FA   LA   LA   LA  ...  LA   LA   LA   MTP
           ───  ───  ───  ───  ───  ───  ───  ───      ───  ───  ───  ────
           全   线   线   线   全   线   线   线       线   线   线  MTP
           注   性   性   性   注   性   性   性       性   性   性  预测
           意   注   注   注   意   注   注   注       注   注   注   头
           力   意   意   意   力   意   意   意       意   意   意

FA = 全注意力 (Full Attention)  → build_layer_attn
LA = 线性注意力 (Linear Attn/SSM) → build_layer_attn_linear
MTP = Multi-Token Prediction 层 → graph_mtp

模式: 每隔 full_attn_interval=4 层插入一个全注意力层，
      其余层使用 SSM 线性注意力（循环状态保持）
```

---

## 8. 数据流汇总

```
Token IDs
    │
    ▼
┌──────────────┐
│  词嵌入查找   │ ────── tok_embd
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                  主解码器循环 (N 层)                           │
│                                                              │
│  inpSA = 输入                                                  │
│       │                                                      │
│       ▼                                                      │
│  attn_norm(RMS) ──────────────────────────────────────────┐  │
│       │                                                     │  │
│       ├── is_recurrent=否 ──→ build_layer_attn ─────────┐   │  │
│       │                    (全注意力)                  │   │  │
│       │                                                 │   │  │
│       └── is_recurrent=是 ──→ build_layer_attn_linear──┼───┘  │
│                            (SSM 线性注意力)               │      │
│                             │                            │      │
│                             │ <── 循环状态 mctx (r_l, s_l)│      │
│       │                     │                             │      │
│       ▼                     ▼                             │      │
│  + inpSA (残差)           + inpSA (残差)                   │      │
│       │                     │                             │      │
│       ▼                     ▼                             │      │
│  attn_post_norm(RMS)  attn_post_norm(RMS)                  │      │
│       │                     │                             │      │
│       ▼                     ▼                             │      │
│  build_layer_ffn ───────── build_layer_ffn                  │      │
│       │                     │                             │      │
│       ├── MoE: softmax 选择 Top-K 专家                      │      │
│       │     Σ(weight_i * FFN_i(x))                        │      │
│       │                     │                             │      │
│       └── + Shared Expert: sigmoid(gate) * FFN(x) ────────┘      │
│                      │                                        │
│                      ▼                                        │
│                 + ffn_residual (残差)                         │
│                      │                                        │
│                      ▼                                        │
│               下一层的 inpSA ──────────────────────────────────┘
│                      │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│  最终归一化 (output_norm, RMS)          │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  LM Head (output weight)              │
│  [n_embd, n_vocab]                   │
└──────────────────┬───────────────────┘
                   │
                   ▼
               t_logits
                   │
                   ├──→ 主模型自回归生成
                   │
                   └──→ MTP 层 (可选)
                         │
                         ├─→ 预测下一个 token
                         └─→ 输出 h_pre_norm (AR 草稿播种)
```

---

## 图例

| 符号 | 含义 |
|------|------|
| `──→` | 数据流向 |
| `─┤` | 条件分支（YES/NO） |
| `[阶段 N]` | 流水线阶段分组 |
| 虚线框 | 循环体 / 条件分支 |
| `build_*` | llama.cpp 中的计算图构建函数 |
| `ggml_*` | GGML 底层张量运算 |
| `RMS` | RMSNorm 归一化 |
| `SSM` | State Space Model（状态空间模型） |
| `MoE` | Mixture of Experts（专家混合） |
| `MTP` | Multi-Token Prediction（多 token 预测） |
| `IMRoPE` | Improved Multi-Head Rotary Position Embedding |
