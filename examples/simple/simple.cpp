/**
 * @file simple.cpp
 * @brief llama.cpp 最简示例程序
 *
 * 演示 llama.cpp 文本生成的基本流程：
 *   1. 解析命令行参数
 *   2. 加载 GGUF (GGML Unified Format) 格式的模型文件
 *   3. 将提示词 (prompt) 分词 (tokenize) 为 token 序列
 *   4. 初始化推理上下文 (context) 和采样器 (sampler)
 *   5. 逐 token 推理并将输出打印到标准输出
 *   6. 打印性能统计信息并释放资源
 *
 * 缩写对照表：
 *   - ggml   -> Graphical Model Machine Learning (ggml 库的名称)
 *   - vocab  -> vocabulary (词表，模型用于 token 与文本相互映射的数据结构)
 *   - ngl    -> number of GPU layers (要卸载到 GPU 的层数)
 *   - n_ctx  -> context size (上下文窗口大小，一次推理能容纳的最大 token 数)
 *   - n_batch   -> number of batch (批量大小，单次 llama_decode 能处理的最大 token 数)
 *   - n_predict -> number of predict (要生成的 token 数量上限)
 *   - n_prompt  -> number of prompt (提示词的 token 数量)
 *   - sparams   -> sampler parameters (采样器参数)
 *   - ctx       -> context (推理上下文，保存推理过程中的中间状态)
 *   - smpl      -> sampler (采样器，决定如何从 logits 中选择下一个 token)
 *   - bos/eos   -> beginning of sentence / end of sentence (句子起始/结束标记)
 *   - eog       -> end of generation (生成结束)
 *   - perf      -> performance (性能统计)
 *   - n_pos     -> number of position (当前位置的 token 计数)
 *   - n_decode  -> number of decode (已解码/生成的 token 数量)
 *   - t_main_start -> time main start (主循环开始时间，微秒)
 *   - t_main_end   -> time main end (主循环结束时间，微秒)
 *   - gguf      -> GGML Unified Format (ggml 统一的模型文件格式)
 *   - argv      -> argument vector (参数向量，命令行参数数组)
 *   - argc      -> argument count (参数计数，命令行参数个数)
 */

#include "llama.h"       // llama.cpp 核心库头文件，提供模型加载、推理、分词等所有 API
#include <clocale>       // C 标准库，用于设置本地化 (setlocale)
#include <cstdio>        // C 标准输入输出 (printf, fprintf, fflush)
#include <cstring>       // C 字符串操作 (strcmp)
#include <string>        // C++ 字符串类
#include <vector>        // C++ 动态数组容器
#include <algorithm>     // std::sort, std::min
#include <cmath>         // std::exp (softmax)
#include <cinttypes>     // PRIu64 等跨平台整数格式化宏
#include <sys/stat.h>    // stat() 检查文件是否存在

// ============================================================================
// 调试宏开关
// ----------------------------------------------------------------------------
//   DEBUG = 1   -> 开启所有调试输出（默认）
//   DEBUG = 0   -> 关闭调试输出，与原始 simple.cpp 行为一致
// 可以通过 -DDEBUG=0 重新编译来关闭调试信息。
// ============================================================================
#ifndef DEBUG
#define DEBUG 1
#endif

#if DEBUG
#define DBG(...)            do { fprintf(stderr, "[DBG] " __VA_ARGS__); fflush(stderr); } while (0)
#define DBG_SECTION(title)  do { fprintf(stderr, "\n[DBG] ====== %s ======\n", title); fflush(stderr); } while (0)
#else
#define DBG(...)            do {} while (0)
#define DBG_SECTION(title)  do {} while (0)
#endif

// 把 enum llama_token_attr 转为可读字符串，便于调试观察 token 类型
static const char * token_attr_to_str(enum llama_token_attr attr) {
    if (attr == LLAMA_TOKEN_ATTR_UNDEFINED)    return "UNDEFINED";
    if (attr & LLAMA_TOKEN_ATTR_UNKNOWN)      return "UNKNOWN";
    if (attr & LLAMA_TOKEN_ATTR_UNUSED)       return "UNUSED";
    if (attr & LLAMA_TOKEN_ATTR_NORMAL)       return "NORMAL";
    if (attr & LLAMA_TOKEN_ATTR_CONTROL)      return "CONTROL";
    if (attr & LLAMA_TOKEN_ATTR_USER_DEFINED) return "USER_DEFINED";
    if (attr & LLAMA_TOKEN_ATTR_BYTE)         return "BYTE";
    if (attr & LLAMA_TOKEN_ATTR_NORMALIZED)   return "NORMALIZED";
    if (attr & LLAMA_TOKEN_ATTR_LSTRIP)       return "LSTRIP";
    if (attr & LLAMA_TOKEN_ATTR_RSTRIP)       return "RSTRIP";
    if (attr & LLAMA_TOKEN_ATTR_SINGLE_WORD)  return "SINGLE_WORD";
    return "(other)";
}

// 把 enum llama_vocab_type 转为可读字符串
static const char * vocab_type_to_str(enum llama_vocab_type t) {
    switch (t) {
        case LLAMA_VOCAB_TYPE_NONE:   return "NONE";
        case LLAMA_VOCAB_TYPE_SPM:    return "SPM (SentencePiece/BPE-LLaMA)";
        case LLAMA_VOCAB_TYPE_BPE:    return "BPE (GPT-2 style)";
        case LLAMA_VOCAB_TYPE_WPM:    return "WPM (WordPiece/BERT)";
        case LLAMA_VOCAB_TYPE_UGM:    return "UGM (Unigram/T5)";
        case LLAMA_VOCAB_TYPE_RWKV:   return "RWKV";
        case LLAMA_VOCAB_TYPE_PLAMO2: return "PLAMO2";
        default:                      return "UNKNOWN";
    }
}

/**
 * @brief 打印用法说明
 * @param argc Argument Count，命令行参数个数
 * @param argv Argument Vector，命令行参数字符串数组
 */
static void print_usage(int argc, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

/**
 * @brief 程序入口点
 *
 * 程序执行流程：
 *   [1] 设置数字本地化格式（保证浮点数输出格式一致）
 *   [2] 解析命令行参数（模型路径、生成 token 数、GPU 层数、提示词）
 *   [3] 加载所有 ggml 后端 (ggml_backend_load_all)
 *   [4] 使用 llama_model_load_from_file 从 GGUF 文件加载模型
 *   [5] 使用 llama_tokenize 将提示词文本转换为 token 数组
 *   [6] 使用 llama_init_from_model 创建推理上下文
 *   [7] 初始化采样器链 (sampler chain)，使用贪心 (greedy) 采样
 *   [8] 将提示词的所有 token 通过 llama_decode 一次性传入模型进行前向传播
 *   [9] 主循环：每次 llama_decode 前向传播 -> 采样新 token -> 打印输出
 *  [10] 采样到 EOS/EOG token 或达到 n_predict 上限时退出循环
 *  [11] 打印性能统计并释放所有资源
 */
int main(int argc, char ** argv) {
    // [1] 设置本地化：确保 printf 输出浮点数时使用 "." 作为小数点分隔符，
    //     避免不同语言环境下输出格式不一致（例如德语用逗号）
    std::setlocale(LC_NUMERIC, "C");

    // [2] 命令行参数默认值定义
    // model_path: GGUF 模型文件的磁盘路径
    std::string model_path;
    // prompt: 输入提示词，默认 "Hello my name is"
    std::string prompt = "Hello my name is";
    // ngl: Number of GPU Layers，将多少层神经网络卸载到 GPU 加速计算 (99 表示全部)
    int ngl = 99;
    // n_predict: Number of Predict，要生成的最大 token 数量
    int n_predict = 32;

    // [2] 解析命令行参数
    // 支持的参数格式：
    //   -m <path>      指定模型 GGUF 文件路径（必选）
    //   -n <n>         指定 n_predict，生成的最大 token 数
    //   -ngl <n>       指定 ngl，卸载到 GPU 的层数
    //   [剩余参数]     作为提示词 (prompt) 处理
    {
        int i = 1;  // 从 argv[1] 开始解析（argv[0] 是程序名）
        for (; i < argc; i++) {
            // 解析 -m：模型文件路径（必选）
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];  // 取出下一个参数作为模型路径
                } else {
                    print_usage(argc, argv); // 参数不完整，打印用法
                    return 1;
                }
            }
            // 解析 -n：Number of predict，要生成的 token 上限
            else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]); // 字符串转整数
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // 解析 -ngl：Number of GPU Layers，卸载到 GPU 的层数
            else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // 遇到未知参数，说明后续都是提示词内容
            else {
                break;
            }
        }

        // 检查是否提供了必需的 -m 参数
        if (model_path.empty()) {
// 没有显式传 -m 时，尝试用本地默认路径（方便调试）
        // 把这条规则集中放在这里：以后想换默认模型，只改这一处即可
        // 顺序很重要：放在前面的会优先被选中；建议把"确定能跑通"的模型放第一位
        static const char * const kDefaultModelPaths[] = {
            "C:\\Users\\zheng\\llm-models\\Gemma-3-1B_Q4_K_M.gguf",
            "C:\\Users\\zheng\\llm-models\\Qwen3-4B-Q4_K_M.gguf",
            "D:\\code\\llama.cpp\\models\\Gemma-3-1B_Q4_K_M.gguf",
            "D:\\code\\llama.cpp\\models\\Qwen3-4B-Q4_K_M.gguf",
            "./Gemma-3-1B_Q4_K_M.gguf",
            "./Qwen3-4B-Q4_K_M.gguf",
            "./model.gguf",
        };

            for (const char * p : kDefaultModelPaths) {
                struct stat st;
                if (stat(p, &st) == 0 && (st.st_mode & S_IFREG)) {
                    model_path = p;
                    fprintf(stderr,
                            "[simple] -m not provided, auto-using default model: %s\n"
                            "[simple] (use -m <path> to override)\n",
                            model_path.c_str());
                    break;
                }
            }

            if (model_path.empty()) {
                fprintf(stderr,
                        "[simple] error: no model file (-m) and no default model found.\n"
                        "[simple] default search paths:\n");
                for (const char * p : kDefaultModelPaths) {
                    fprintf(stderr, "    %s\n", p);
                }
                print_usage(argc, argv);
                return 1;
            }
        }

        // 将剩余的所有参数拼接为提示词 (prompt)
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    DBG_SECTION("Command-line arguments");
    DBG("argv[0]       = %s\n", argv[0]);
    DBG("argc          = %d\n", argc);
    DBG("model_path    = %s\n", model_path.c_str());
    DBG("prompt        = \"%s\" (len=%zu)\n", prompt.c_str(), prompt.size());
    DBG("n_predict     = %d\n", n_predict);
    DBG("ngl (GPU layers) = %d\n", ngl);

    // [3] 加载所有可用的 ggml 后端 (ggml_backend_load_all)
    // 这会根据系统环境自动加载 CPU、CUDA (NVIDIA)、Metal (Apple)、Vulkan 等后端
    ggml_backend_load_all();

    // [4] 模型加载阶段

    // llama_model_load_from_file: 从 GGUF 文件加载模型到内存
    // 返回指向 llama_model 结构的指针，失败时返回 NULL
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // llama_model_get_vocab: 获取模型内置的词表 (vocabulary)
    // vocab 包含 token->文本 和 文本->token 的双向映射，以及特殊 token (EOS/BOS 等)
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // === 模型元信息调试打印 ===
    DBG_SECTION("Model metadata");
    {
        char desc_buf[1024];
        llama_model_desc(model, desc_buf, sizeof(desc_buf));
        DBG("model description : %s\n", desc_buf);

        DBG("model size        : %.2f MiB (n_params=%" PRIu64 ")\n",
                (double) llama_model_size(model) / (1024.0 * 1024.0),
                (uint64_t) llama_model_n_params(model));

        DBG("context size(train)  : %d\n", llama_model_n_ctx_train(model));
        DBG("embedding dim        : in=%d  out=%d\n",
                llama_model_n_embd_inp(model), llama_model_n_embd_out(model));
        DBG("num layers           : %d\n", llama_model_n_layer(model));
        DBG("num heads            : %d (kv_heads=%d)\n",
                llama_model_n_head(model), llama_model_n_head_kv(model));
        DBG("SWA layers           : %d\n", llama_model_n_swa(model));

        DBG("vocab type           : %s (id=%d)\n",
                vocab_type_to_str(llama_vocab_type(vocab)),
                (int) llama_vocab_type(vocab));
        DBG("vocab size           : %d tokens\n", llama_vocab_n_tokens(vocab));
        DBG("BOS / EOS / EOT      : %d / %d / %d\n",
                (int) llama_vocab_bos(vocab),
                (int) llama_vocab_eos(vocab),
                (int) llama_vocab_eot(vocab));
        DBG("add BOS in spec. tok : %s\n", llama_vocab_get_add_bos(vocab) ? "true" : "false");
        DBG("add EOS in spec. tok : %s\n", llama_vocab_get_add_eos(vocab) ? "true" : "false");

        const char * chat_tpl = llama_model_chat_template(model, "default");
        DBG("chat template       : %s\n", chat_tpl ? chat_tpl : "(none)");
    }

    // [5] 分词阶段 (tokenize)

    // llama_tokenize: 将文本字符串转换为 token ID 数组
    // 第一次调用时 buffer=NULL, n_tokens=0，用于查询需要多少个 token（返回负数表示数量）
    // 第 3 个参数 NULL: 不写入任何 token，只计算数量
    // 第 5 个参数 true: 添加特殊 token (如 BOS)；第 6 个参数 true: 特殊处理中文字符
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // 第二次调用时传入预分配的 token 数组，执行实际分词
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
            prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // === 分词结果调试打印 ===
    DBG_SECTION("Tokenization");
    DBG("n_prompt = %d tokens\n", n_prompt);
    DBG("token id sequence : [");
    for (int j = 0; j < n_prompt; ++j) {
        fprintf(stderr, "%s%d", j ? ", " : "", prompt_tokens[j]);
    }
    fprintf(stderr, "]\n");

    DBG("token details:\n");
    for (int j = 0; j < n_prompt; ++j) {
        const llama_token id = prompt_tokens[j];
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        enum llama_token_attr attr = llama_vocab_get_attr(vocab, id);
        if (n < 0) { buf[0] = '\0'; n = 0; }
        // 把 buf 中的非可打印字符转义，便于观察
        std::string repr;
        for (int k = 0; k < n; ++k) {
            unsigned char c = (unsigned char) buf[k];
            if (c == '\n') repr += "\\n";
            else if (c == '\r') repr += "\\r";
            else if (c == '\t') repr += "\\t";
            else if (c < 0x20 || c == 0x7f) {
                char tmp[8]; snprintf(tmp, sizeof(tmp), "\\x%02x", c); repr += tmp;
            } else {
                repr.push_back((char) c);
            }
        }
        DBG("  [%2d] id=%6d  attr=0x%02x(%-12s) text=\"%s\"\n",
                j, (int) id, (unsigned) attr, token_attr_to_str(attr), repr.c_str());
    }

    // [6] 推理上下文初始化

    // llama_context_default_params: 获取上下文的默认参数
    // ctx_params: 控制推理上下文的创建方式（内存分配、KV 缓存大小等）
    llama_context_params ctx_params = llama_context_default_params();

    // n_ctx: Context size，上下文窗口大小，决定单次推理能处理的最大 token 数
    // 这里设置为 n_prompt + n_predict - 1：
    //   - 提示词 token 占用 n_prompt
    //   - 生成部分最多占用 n_predict - 1（新生成的 token）
    //   - 总共需要 n_prompt + n_predict - 1 个位置
    ctx_params.n_ctx = n_prompt + n_predict - 1;

    // n_batch: Number of batch，单次 llama_decode 调用能处理的最大 token 数
    // 这里设置为 n_prompt，即把整个提示词放在一个批次中处理
    ctx_params.n_batch = n_prompt;

    // no_perf: 是否禁用性能计数器。false 表示启用，以便后续打印统计信息
    ctx_params.no_perf = false;

    DBG_SECTION("Context params");
    DBG("n_ctx       = %u\n", ctx_params.n_ctx);
    DBG("n_batch     = %u\n", ctx_params.n_batch);
    DBG("n_ubatch    = %u\n", ctx_params.n_ubatch);
    DBG("n_seq_max   = %u\n", ctx_params.n_seq_max);
    DBG("n_threads   = ctx=%d batch=%d\n",
            ctx_params.n_threads, ctx_params.n_threads_batch);
    DBG("no_perf     = %s\n", ctx_params.no_perf ? "true" : "false");
    DBG("flash_attn  = %s\n", llama_flash_attn_type_name(ctx_params.flash_attn_type));
    DBG("op_offload  = %s\n", ctx_params.op_offload ? "true" : "false");

    // llama_init_from_model: 使用已加载的模型初始化推理上下文
    // ctx 保存了 KV 缓存 (Key-Value cache) 和注意力机制的中间结果
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    // === 上下文运行时信息 ===
    DBG_SECTION("Context runtime");
    DBG("llama_n_ctx      = %u\n", llama_n_ctx(ctx));
    DBG("llama_n_ctx_seq  = %u\n", llama_n_ctx_seq(ctx));
    DBG("llama_n_batch    = %u\n", llama_n_batch(ctx));

    // [7] 采样器初始化 (sampler)

    // llama_sampler_chain_default_params: 获取采样器链的默认参数
    // sparams: Sampler Parameters，控制采样器的通用行为
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;  // 启用采样器性能统计

    // llama_sampler_chain_init: 创建一个采样器链（可以串联多个采样策略）
    // smpl: Sampler，本例中只使用最简单的贪心采样 (greedy sampling)
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // llama_sampler_chain_add: 向采样器链添加一个具体的采样器
    // llama_sampler_init_greedy: 贪心采样，每步直接选择概率最高的 token
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    DBG_SECTION("Sampler");
    DBG("sampler chain : %s\n", llama_sampler_name(smpl));

    // [8] 打印提示词（token by token，将每个 token 转回文本输出）
    for (auto id : prompt_tokens) {
        char buf[128];
        // llama_token_to_piece: 将单个 token ID 转换回文本片段
        // 返回值 n 为写入 buf 的字节数（负数表示失败）
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // [8.5] 准备第一批数据 (prompt batch)

    // llama_batch_get_one: 创建一个只包含一个 token 序列的批次结构
    // 这里将整个 prompt_tokens 作为一个批次传入
    // batch: 数据批次结构，包含所有要处理的 token 的位置信息
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // 检查模型是否使用编码器-解码器架构（如 T5 等 encoder-decoder 模型）
    if (llama_model_has_encoder(model)) {
        // llama_encode: 对编码器部分进行前向传播
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        // llama_model_decoder_start_token: 获取解码器的起始 token（如翻译任务中目标语言的 BOS）
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        // 如果模型没有专门的解码器起始 token，则回退到词表的 BOS token
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        // 用解码器起始 token 重新构建批次，准备进入解码阶段
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // [9] 主推理循环

    // ggml_time_us: 获取当前时间（微秒），用于计时
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;         // Number of Decode，已成功生成的 token 计数
    llama_token new_token_id; // 新采样出的 token ID

    // 是否在每一步打印 logits top-k（默认开），关闭可减少输出量
    const bool dbg_print_logits_topk = true;
    const int  dbg_topk              = 5;

    DBG_SECTION("Main loop start");
    DBG("loop exit condition : n_pos + batch.n_tokens < n_prompt(%d) + n_predict(%d)\n",
            n_prompt, n_predict);

    // 循环条件：当前位置 n_pos 加上当前批次的 token 数 < 总需要的 token 数
    // 循环会在以下情况退出：
    //   - 采样到 EOS/EOG token（生成自然结束）
    //   - 到达了 n_prompt + n_predict 的总 token 数量限制
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        const int step_decode = n_decode; // 当前是第几步（0=prompt prefill）
        const bool is_prefill = (batch.n_tokens > 1);

        // ---- decode 前：打印 batch 内容 ----
        DBG_SECTION("Before llama_decode()");
        DBG("step=%d (%s), n_pos=%d\n",
                step_decode, is_prefill ? "PREFILL(prompt)" : "DECODE(generate)", n_pos);
        DBG("batch.n_tokens = %d\n", batch.n_tokens);
        DBG("batch.token ids = [");
        for (int32_t t = 0; t < batch.n_tokens; ++t) {
            fprintf(stderr, "%s%d", t ? ", " : "", batch.token[t]);
        }
        fprintf(stderr, "]\n");
        if (batch.pos) {
            DBG("batch.pos  = [");
            for (int32_t t = 0; t < batch.n_tokens; ++t) {
                fprintf(stderr, "%s%d", t ? ", " : "", batch.pos[t]);
            }
            fprintf(stderr, "]\n");
        }
        if (batch.n_seq_id) {
            DBG("batch.seq_id = [");
            for (int32_t t = 0; t < batch.n_tokens; ++t) {
                fprintf(stderr, "%s%d", t ? ", " : "", batch.seq_id[t][0]);
            }
            fprintf(stderr, "]\n");
        }
        if (batch.logits) {
            DBG("batch.logits(output flag) = [");
            for (int32_t t = 0; t < batch.n_tokens; ++t) {
                fprintf(stderr, "%s%d", t ? ", " : "", (int) batch.logits[t]);
            }
            fprintf(stderr, "]\n");
        }

        // llama_decode: 执行一次模型前向传播
        //   - 对 prompt batch：处理整个提示词，计算出最后一个 token 对应的隐藏状态
        //   - 对单个 token batch：计算新 token 的 logits（下一个 token 的概率分布）
        const auto t_decode_start = ggml_time_us();
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        const auto t_decode_end = ggml_time_us();

        // 累加已处理的 token 位置
        n_pos += batch.n_tokens;

        // ---- decode 后：打印耗时 + logits top-k ----
        DBG("llama_decode() finished, took %.2f ms (%.1f tokens/s)\n",
                (t_decode_end - t_decode_start) / 1000.0f,
                batch.n_tokens * 1000000.0f / (float)(t_decode_end - t_decode_start + 1));

        if (dbg_print_logits_topk) {
            // 取最后一个位置的 logits (i = batch.n_tokens - 1)
            const int32_t i = batch.n_tokens - 1;
            const float * logits = llama_get_logits_ith(ctx, i);
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            if (logits == nullptr) {
                DBG("logits=NULL (no logits at this position)\n");
            } else {
                // 计算 logits 统计：min/max，并做 softmax 求 top-k 概率
                float lmin = logits[0], lmax = logits[0], lsum = logits[0];
                for (int32_t v = 1; v < n_vocab; ++v) {
                    if (logits[v] < lmin) lmin = logits[v];
                    if (logits[v] > lmax) lmax = logits[v];
                    lsum += logits[v];
                }
                const float mean = lsum / (float) n_vocab;
                DBG("logits[%d] : n_vocab=%d  min=%.4f  max=%.4f  mean=%.4f\n",
                        i, n_vocab, lmin, lmax, mean);

                // softmax 概率
                std::vector<float> probs(n_vocab);
                float pmax = lmax;
                float psum = 0.0f;
                for (int32_t v = 0; v < n_vocab; ++v) {
                    probs[v] = std::exp(logits[v] - pmax);
                    psum += probs[v];
                }
                for (int32_t v = 0; v < n_vocab; ++v) probs[v] /= psum;

                // 取 top-k
                std::vector<int32_t> idx(n_vocab);
                for (int32_t v = 0; v < n_vocab; ++v) idx[v] = v;
                int k = std::min(dbg_topk, n_vocab);
                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });

                DBG("logits top-%d (sorted by probability desc):\n", k);
                for (int j = 0; j < k; ++j) {
                    int32_t tid = idx[j];
                    char tbuf[256];
                    int tn = llama_token_to_piece(vocab, tid, tbuf, sizeof(tbuf), 0, false);
                    if (tn < 0) { tbuf[0] = '\0'; tn = 0; }
                    std::string repr;
                    for (int p = 0; p < tn; ++p) {
                        unsigned char c = (unsigned char) tbuf[p];
                        if (c == '\n') repr += "\\n";
                        else if (c == '\r') repr += "\\r";
                        else if (c == '\t') repr += "\\t";
                        else if (c < 0x20 || c == 0x7f) {
                            char tmp[8]; snprintf(tmp, sizeof(tmp), "\\x%02x", c); repr += tmp;
                        } else {
                            repr.push_back((char) c);
                        }
                    }
                    DBG("  #%d  id=%6d  logit=% .4f  prob=%.4f  text=\"%s\"\n",
                            j + 1, tid, logits[tid], probs[tid], repr.c_str());
                }
            }
        }

        // [9.1] 采样下一个 token
        {
            // llama_sampler_sample: 根据模型输出的 logits，从采样器链中采样一个 token
            // 第三个参数 -1 表示使用当前上下文中最后一个 token 的位置
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            char tbuf[256];
            int tn = llama_token_to_piece(vocab, new_token_id, tbuf, sizeof(tbuf), 0, true);
            if (tn < 0) { tbuf[0] = '\0'; tn = 0; }
            std::string s(tbuf, tn);
            enum llama_token_attr attr = llama_vocab_get_attr(vocab, new_token_id);

DBG_SECTION("Sample");
        DBG("sampler chain      : %s\n", llama_sampler_name(smpl));
        DBG("sampled token      : id=%d, attr=0x%x(%s), text=\"%s\"\n",
                (int) new_token_id, (unsigned) attr,
                token_attr_to_str(attr), s.c_str());
        DBG("is_eog             : %s\n",
                llama_vocab_is_eog(vocab, new_token_id) ? "true" : "false");

        // llama_vocab_is_eog: check whether the sampled token is End-Of-Generation
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            DBG("--> hit EOG, exit loop\n");
            break;  // generation finished, leave loop
            }

            printf("%s", s.c_str());
            fflush(stdout);  // 立即刷新输出缓冲区，确保实时显示

            // 为下一轮推理准备批次：只包含刚采样出的新 token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;  // 已解码 token 计数 +1
        }
    }

    printf("\n");

    // [10] 性能统计

    const auto t_main_end = ggml_time_us();

    // 打印总耗时和生成速度 (tokens per second)
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode,
            (t_main_end - t_main_start) / 1000000.0f,  // 微秒转秒
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));  // tokens / 秒

    fprintf(stderr, "\n");

    // llama_perf_sampler_print: 打印采样器内部的性能统计信息
    //   包含采样次数、各采样策略的执行时间等
    llama_perf_sampler_print(smpl);

    // llama_perf_context_print: 打印推理上下文的性能统计信息
    //   包含 KV 缓存命中率、各层计算耗时、内存使用情况等
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    DBG_SECTION("Summary");
    DBG("n_decode             = %d\n", n_decode);
    {
        const double elapsed_s = (t_main_end - t_main_start) / 1000000.0;
        DBG("total time           = %.2f s\n", elapsed_s);
        if (n_decode > 0 && elapsed_s > 0.0) {
            DBG("generation speed     = %.2f tokens/s\n", n_decode / elapsed_s);
        } else {
            DBG("generation speed     = n/a\n");
        }
    }

    // [11] 资源释放（按照创建顺序的逆序释放）
    llama_sampler_free(smpl);  // 释放采样器
    llama_free(ctx);           // 释放推理上下文
    llama_model_free(model);   // 释放模型

    return 0;
}