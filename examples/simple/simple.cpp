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
 *   - ggml  -> Graphical Model Machine Learning（ggml 库的名称）
 *   - vocab -> vocabulary（词表，模型用于 token 与文本相互映射的数据结构）
 *   - ngl   -> number of GPU layers（要卸载到 GPU 的层数）
 *   - n_ctx -> context size（上下文窗口大小，一次推理能容纳的最大 token 数）
 *   - n_batch -> number of batch（批量大小，单次 llama_decode 能处理的最大 token 数）
 *   - n_predict -> number of predict（要生成的 token 数量上限）
 *   - n_prompt -> number of prompt（提示词的 token 数量）
 *   - sparams -> sampler parameters（采样器参数）
 *   - ctx    -> context（推理上下文，保存推理过程中的中间状态）
 *   - smpl   -> sampler（采样器，决定如何从 logits 中选择下一个 token）
 *   - bos/eos -> beginning of sentence / end of sentence（句子起始/结束标记）
 *   - eog    -> end of generation（生成结束）
 *   - perf   -> performance（性能统计）
 *   - n_pos  -> number of position（当前位置的 token 计数）
 *   - n_decode -> number of decode（已解码/生成的 token 数量）
 *   - t_main_start -> time main start（主循环开始时间，微秒）
 *   - t_main_end   -> time main end（主循环结束时间，微秒）
 *   - gguf   -> GGML Unified Format（ggml 统一的模型文件格式）
 *   - argv   -> argument vector（参数向量，命令行参数数组）
 *   - argc   -> argument count（参数计数，命令行参数个数）
 */

#include "llama.h"       // llama.cpp 核心库头文件，提供模型加载、推理、分词等所有 API
#include <clocale>       // C 标准库，用于设置本地化（setlocale）
#include <cstdio>        // C 标准输入输出（printf, fprintf, fflush）
#include <cstring>       // C 字符串操作（strcmp）
#include <string>        // C++ 字符串类
#include <vector>        // C++ 动态数组容器

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
 *   [3] 加载所有 ggml 后端（ggml_backend_load_all）
 *   [4] 使用 llama_model_load_from_file 从 GGUF 文件加载模型
 *   [5] 使用 llama_tokenize 将提示词文本转换为 token 数组
 *   [6] 使用 llama_init_from_model 创建推理上下文
 *   [7] 初始化采样器链（sampler chain），使用纯贪婪（greedy）采样
 *   [8] 将提示词的所有 token 通过 llama_decode 一次性送入模型进行前向传播
 *   [9] 主循环：每次 llama_decode 前向传播 -> 采样新 token -> 打印输出
 *  [10] 采样到 EOS/EOG token 或达到 n_predict 上限时退出循环
 *  [11] 打印性能统计并释放所有资源
 */
int main(int argc, char ** argv) {
    // [1] 设置本地化：确保 printf 输出浮点数时使用 "." 作为小数点分隔符，
    //     避免不同语言环境下输出格式不一致（如德语用逗号）
    std::setlocale(LC_NUMERIC, "C");

    // [2] 命令行参数默认值定义
    // model_path: GGUF 模型文件的磁盘路径
    std::string model_path;
    // prompt: 输入提示词，即"Hello my name is"，作为默认提示
    std::string prompt = "Hello my name is";
    // ngl: Number of GPU Layers，将多少层神经网络卸载到 GPU 加速计算（99 表示全部）
    int ngl = 99;
    // n_predict: Number of Predict，要生成的最多 token 数量
    int n_predict = 32;

    // [2] 解析命令行参数
    // 支持的参数格式：
    //   -m <path>      指定模型 GGUF 文件路径（必选）
    //   -n <n>         指定 n_predict，生成的最大 token 数
    //   -ngl <n>       指定 ngl，卸载到 GPU 的层数
    //   [剩余参数]     作为提示词（prompt）处理
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
            print_usage(argc, argv);
            return 1;
        }

        // 将剩余的所有参数拼接为提示词（prompt）
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // [3] 加载所有可用的 ggml 后端（ggml_backend_load_all）
    // 这会根据系统环境自动加载 CPU、CUDA（NVIDIA）、Metal（Apple）、Vulkan 等后端
    ggml_backend_load_all();

    // [4] 模型加载阶段

    // llama_model_default_params: 获取模型的默认参数
    // model_params: 模型加载参数结构体，控制模型如何加载到内存/显存
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;  // n_gpu_layers: 设置将多少层卸载到 GPU

    // llama_model_load_from_file: 从 GGUF 文件加载模型到内存
    // 返回指向 llama_model 结构的指针，失败时返回 NULL
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // llama_model_get_vocab: 获取模型内置的词表（vocabulary）
    // vocab 包含 token->文本 和 文本->token 的双向映射，以及特殊 token（EOS/BOS 等）
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // [5] 分词阶段（tokenize）

    // llama_tokenize: 将文本字符串转换为 token ID 数组
    // 第一次调用时 buffer=NULL, n_tokens=0，用于查询需要多少个 token（返回负数表示数量）
    // 第4个参数 NULL: 不写入任何 token，只计算数量
    // 第6个参数 true:  添加特殊 token（如 BOS）；第7个参数 true: 特殊处理中文字符
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // 第二次调用时传入预分配的 token 数组，执行实际分词
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
            prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // [6] 推理上下文初始化

    // llama_context_default_params: 获取上下文的默认参数
    // ctx_params: 控制推理上下文的创建方式（内存分配、KV 缓存大小等）
    llama_context_params ctx_params = llama_context_default_params();

    // n_ctx: Context size，上下文窗口大小，决定单次推理能处理的最大 token 数
    // 这里设置为 n_prompt + n_predict - 1：
    //   - 提示词 token 占用 n_prompt
    //   - 生成部分最多占用 n_predict - 1（新生成的 token）
    //   - 总共需要 n_prompt + n_predict - 1 个槽位
    ctx_params.n_ctx = n_prompt + n_predict - 1;

    // n_batch: Number of batch，单次 llama_decode 调用能处理的最大 token 数
    // 这里设置为 n_prompt，即把整个提示词放在一个批次中处理
    ctx_params.n_batch = n_prompt;

    // no_perf: 是否禁用性能计数器。false 表示启用，以便后续打印统计信息
    ctx_params.no_perf = false;

    // llama_init_from_model: 使用已加载的模型初始化推理上下文
    // ctx 保存了 KV 缓存（Key-Value cache）和注意力机制的中间结果
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    // [7] 采样器初始化（sampler）

    // llama_sampler_chain_default_params: 获取采样器链的默认参数
    // sparams: Sampler Parameters，控制采样器的通用行为
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;  // 启用采样器性能统计

    // llama_sampler_chain_init: 创建一个采样器链（可以串联多个采样策略）
    // smpl: Sampler，本例中只使用最简单的贪婪采样（greedy sampling）
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // llama_sampler_chain_add: 向采样器链添加一个具体的采样器
    // llama_sampler_init_greedy: 贪婪采样，每步直接选择概率最高的 token
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

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

    // [8.5] 准备第一批数据（prompt batch）

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

    // 循环条件：当前位置 n_pos 加上当前批次的 token 数 < 总需要的 token 数
    // 循环会在以下情况退出：
    //   - 采样到 EOS/EOG token（生成自然结束）
    //   - 达到了 n_prompt + n_predict 的总 token 数量限制
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {

        // llama_decode: 执行一次模型前向传播
        //   - 对 prompt batch：处理整个提示词，计算出最后一个 token 对应的隐状态
        //   - 对单个 token batch：计算新 token 的 logits（下一个 token 的概率分布）
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // 累加已处理的 token 位置
        n_pos += batch.n_tokens;

        // [9.1] 采样下一个 token
        {
            // llama_sampler_sample: 根据模型输出的 logits，从采样器链中采样一个 token
            // 第三个参数 -1 表示使用当前上下文中的最后一个 token 的位置
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // llama_vocab_is_eog: 检查是否到达生成结束标记（End Of Generation）
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;  // 生成完毕，退出循环
            }

            // llama_token_to_piece: 将采样出的 token 转换为文本片段
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);  // 立即刷新输出缓冲区，确保实时显示

            // 为下一轮推理准备批次：只包含刚刚采样出的新 token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;  // 已解码 token 计数 +1
        }
    }

    printf("\n");

    // [10] 性能统计

    const auto t_main_end = ggml_time_us();

    // 打印总耗时和生成速度（tokens per second）
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

    // [11] 资源释放（按照创建顺序的逆序释放）
    llama_sampler_free(smpl);  // 释放采样器
    llama_free(ctx);           // 释放推理上下文
    llama_model_free(model);   // 释放模型

    return 0;
}
