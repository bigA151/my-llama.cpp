/**
 * llama.cpp 公共 API 实现
 *
 * 缩写词展开说明:
 *   llama  = "Large Language Model" (大语言模型)
 *   attn   = "attention" (注意力机制)
 *   mmap   = "memory-mapped file" (内存映射文件)
 *   mlock  = "memory lock" (内存锁)
 *   gpu    = "Graphics Processing Unit" (图形处理器)
 *   rpc    = "Remote Procedure Call" (远程过程调用)
 *   numa   = "Non-Uniform Memory Access" (非统一内存访问)
 *   buft   = "buffer type" (缓冲区类型)
 *   kv     = "key-value" (键值对)
 *   arch   = "architecture" (架构)
 *   hparams = "hyperparameters" (超参数)
 *   vocab  = "vocabulary" (词表)
 *   ctx    = "context" (上下文)
 *   tmpl   = "template" (模板)
 *   ass    = "assistant" (助手)
 *   devs   = "devices" (设备)
 *   params = "parameters" (参数)
 *   impl   = "implementation" (实现)
 *   ud     = "user data" (用户数据)
 */

#include "llama.h"

#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

/**
 * llama_flash_attn_type_name
 * flash attention (FlashAttention, 快速注意力) 类型名称
 * flash_attn_type: FlashAttention 类型枚举
 * 返回: 对应类型的字符串名称 ("auto" / "disabled" / "enabled")
 */
const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

/**
 * llama_sampler_chain_default_params
 * sampler chain (采样器链) 默认参数
 * 返回: 默认的采样器链参数结构体
 *
 * 采样器链用于将多个采样策略组合起来使用
 */
struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

/**
 * llama_max_devices
 * maximum devices (最大设备数量)
 * 返回: 支持的最大计算设备数量
 */
size_t llama_max_devices(void) {
    return 16;
}

/**
 * llama_max_tensor_buft_overrides
 * maximum tensor buffer type overrides (张量缓冲区类型覆盖的最大数量)
 * 返回: 允许的张量缓冲区类型覆盖的最大条目数
 */
size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

/**
 * llama_supports_mmap
 * supports memory-mapped file (是否支持内存映射文件)
 * 返回: 如果后端支持 mmap 则返回 true
 */
bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

/**
 * llama_supports_mlock
 * supports memory lock (是否支持内存锁定)
 * 返回: 如果后端支持 mlock (锁定内存防止换出) 则返回 true
 */
bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

/**
 * llama_supports_gpu_offload
 * supports GPU (Graphics Processing Unit, 图形处理器) offload (卸载)
 * 返回: 如果系统支持将计算卸载到 GPU 则返回 true
 *
 * 检查 GPU、集成 GPU 或 RPC 后端是否可用
 */
bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

/**
 * llama_supports_rpc
 * supports RPC (Remote Procedure Call, 远程过程调用)
 * 返回: 如果 RPC 后端已加载则返回 true
 */
bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

/**
 * llama_backend_init
 * backend (后端) initialize (初始化)
 * 初始化 ggml 后端和时间模块
 * 需要在调用其他 llama 函数前调用
 */
void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

/**
 * llama_numa_init
 * NUMA (Non-Uniform Memory Access, 非统一内存访问) initialize
 * numa: NUMA 策略配置
 *
 * 在支持 NUMA 的系统上优化内存分配，提高多插槽 CPU 的内存访问效率
 */
void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

/**
 * llama_backend_free
 * backend (后端) free (释放)
 * 释放所有后端相关的资源
 */
void llama_backend_free(void) {
    ggml_quantize_free();
}

/**
 * llama_time_us
 * time (时间) in microseconds (微秒)
 * 返回: 自某个固定起点以来的微秒数
 */
int64_t llama_time_us(void) {
    return ggml_time_us();
}

/**
 * llama_prepare_model_devices
 * prepare model (模型) devices (设备) — 准备模型运行所需的计算设备
 * params: 模型参数，包含设备选择和分片策略
 * model: 待配置设备的模型实例
 *
 * 返回: 成功返回 true，失败返回 false
 *
 * 根据 split_mode 将模型张量分配到合适的设备上:
 *   - LLAMA_SPLIT_MODE_NONE: 仅使用 main_gpu
 *   - LLAMA_SPLIT_MODE_LAYER: 按层分片到多个 GPU
 *   - LLAMA_SPLIT_MODE_TENSOR: 按张量维度分片 (需要 Meta 设备聚合)
 */
// returns true on success
static bool llama_prepare_model_devices(const llama_model_params & params, llama_model * model) {
    // create list of devices to use with this model
    if (params.devices) {
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            size_t n_devs = 0;
            while (params.devices[n_devs]) {
                n_devs++;
            }
            if (n_devs == 0) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, n_devs);
            for (size_t i = 0; i < n_devs; ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s\n", __func__, i, ggml_backend_dev_name(params.devices[i]));
            }
            model->get_split_state_ud.n_devices = n_devs;
            model->get_split_state_ud.model = model;
            model->devices.push_back({
                true, ggml_backend_meta_device(
                params.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
                model->devices.push_back({false, *dev});
            }
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<llama_device> gpus;
        std::vector<llama_device> igpus;
        std::vector<llama_device> rpc_servers;

        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(ggml_backend_dev_count());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
                    LLAMA_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    continue;
                }
                devs.push_back(dev);
            }
            if (devs.empty()) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }

            LLAMA_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, devs.size());
            for (size_t i = 0; i < devs.size(); ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, i, ggml_backend_dev_name(devs[i]), ggml_backend_dev_description(devs[i]));
            }

            GGML_ASSERT(!devs.empty());
            model->get_split_state_ud.n_devices = devs.size();
            model->get_split_state_ud.model     = model;
            gpus.push_back({
                true, ggml_backend_meta_device(
                devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_servers.push_back({false, dev});
                        } else {
                            // check if there is already a GPU with the same device id
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(dev, &props);
                            auto it = std::find_if(gpus.begin(), gpus.end(), [&props](const llama_device & d) {
                                ggml_backend_dev_props d_props;
                                ggml_backend_dev_get_props(d.dev, &d_props);
                                if (props.device_id && d_props.device_id) {
                                    return strcmp(props.device_id, d_props.device_id) == 0;
                                }
                                return false;
                            });

                            if (it != gpus.end()) {
                                LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                        __func__,
                                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                        props.device_id ? props.device_id : "unknown id",
                                        ggml_backend_dev_name(it->dev), ggml_backend_dev_description(it->dev));
                            } else {
                                gpus.push_back({false, dev});
                            }
                        }
                        break;
                    }

                    case GGML_BACKEND_DEVICE_TYPE_IGPU:
                        igpus.push_back({false, dev});
                        break;
                    case GGML_BACKEND_DEVICE_TYPE_META:
                        GGML_ABORT("fatal error");
                }
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no other devices were found
        if (model->devices.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                return false;
            }
            llama_device main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (const auto & dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    return true;
}

/**
 * llama_model_load
 * model (模型) load (加载)
 *
 * metadata: GGUF 元数据上下文
 * set_tensor_data: 设置张量数据的回调函数
 * set_tensor_data_ud: set_tensor_data 的 user data (用户数据)
 * fname: file name (文件名)
 * splits: 分片路径列表 (用于分布式模型)
 * file: 文件指针
 * params: model parameters (模型参数)
 *
 * 返回: <状态码, 模型指针>
 *   0  = 成功
 *  -1  = 加载错误
 *  -2  = 通过 progress_callback 取消
 *
 * 加载流程: 解析元数据 -> 创建模型对象 -> 加载超参数 -> 加载词表 -> 加载张量
 */
// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.use_mmap, params.use_direct_io,
            params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

/**
 * llama_model_load_from_file_impl
 * model load from file implementation (从文件加载模型的实现)
 *
 * metadata: GGUF 元数据 (可选)
 * set_tensor_data: 设置张量数据的回调
 * set_tensor_data_ud: set_tensor_data 的 user data
 * path_model: 模型文件路径
 * splits: 分片路径列表
 * file: 已打开的文件指针 (可选)
 * params: model parameters (模型参数)
 *
 * 注意: metadata / path_model / file 三者必须恰好有一个被指定
 */
static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}

/**
 * llama_model_init_from_user
 * model initialize from user (从用户提供的元数据初始化模型)
 *
 * 通过用户传入的元数据上下文 (而非文件路径) 初始化模型
 * metadata: GGUF 元数据上下文 (必须非空)
 * set_tensor_data: 设置张量数据的回调
 * set_tensor_data_ud: set_tensor_data 的 user data
 * params: model parameters (模型参数)
 */
struct llama_model * llama_model_init_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params) {
    GGML_ASSERT(metadata != nullptr);
    std::string path_model;
    std::vector<std::string> splits = {};
    params.use_mmap = false;
    params.use_extra_bufts = false;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, /*file*/ nullptr, params);
}

/**
 * llama_load_model_from_file
 * load model from file (从文件加载模型) — 已废弃，请使用 llama_model_load_from_file
 *
 * path_model: 模型文件路径
 * params: model parameters (模型参数)
 */
// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

/**
 * llama_model_load_from_file
 * model load from file (从文件加载模型)
 *
 * path_model: 模型文件路径
 * params: model parameters (模型参数)
 */
struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

/**
 * llama_model_load_from_splits
 * model load from splits (从分片文件列表加载模型)
 *
 * 支持从多个分片文件 (如 sharded model) 加载一个分布式模型
 * paths: 分片文件路径数组
 * n_paths: 分片文件数量
 * params: model parameters (模型参数)
 */
struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, splits.front(), splits, /*file*/ nullptr, params);
}

/**
 * llama_model_load_from_file_ptr
 * model load from file pointer (从文件指针加载模型)
 *
 * 通过已打开的 FILE* 加载模型，而非通过文件路径
 * file: 已打开的文件指针
 * params: model parameters (模型参数)
 */
struct llama_model * llama_model_load_from_file_ptr(FILE * file, struct llama_model_params params) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }
    std::string path_model;
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, file, params);
}

/**
 * llama_model_save_to_file
 * model save to file (将模型保存到文件)
 *
 * model: 要保存的模型
 * path_model: 目标文件路径 (GGUF 格式)
 */
void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates (聊天模板)
//

/**
 * llama_chat_apply_template
 * chat apply template (将聊天消息应用模板格式化)
 *
 * tmpl: template (模板) 字符串，可为 nullptr (默认使用 chatml)
 * chat: 聊天消息数组
 * n_msg: 消息数量 (number of messages)
 * add_ass: 是否添加 assistant (助手) 前缀标记
 * buf: 输出缓冲区
 * length: buf 的最大长度
 *
 * 返回: 格式化后的字符串长度，失败返回 -1
 */
int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split (模型分片)
//

/**
 * llama_split_path
 * split path (生成分片文件的完整路径)
 *
 * 将模型路径前缀与分片编号组合，生成标准格式的分片路径
 * 格式: {path_prefix}-{split_no:05d}-of-{split_count:05d}.gguf
 *
 * split_path: 输出缓冲区，接收生成的路径
 * maxlen: split_path 缓冲区的最大长度
 * path_prefix: 路径前缀 (不含分片编号部分)
 * split_no: 当前分片编号 (从 0 开始)
 * split_count: 总分片数量
 *
 * 返回: 写入的字符数，失败返回 0
 */
int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

/**
 * llama_split_prefix
 * split prefix (从分片路径中提取路径前缀)
 *
 * 从分片文件路径中提取出不含分片编号的后缀部分
 *
 * split_prefix: 输出缓冲区，接收提取出的前缀
 * maxlen: split_prefix 缓冲区的最大长度
 * split_path: 分片文件路径 (如 model-00001-of-00003.gguf)
 * split_no: 分片编号 (从 0 开始)
 * split_count: 总分片数量
 *
 * 返回: 写入的字符数，失败返回 0
 */
int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

/**
 * llama_print_system_info
 * print system info (打印系统信息)
 *
 * 返回: 包含所有已加载后端及其特性的字符串
 *
 * 遍历所有已注册的 ggml 后端，收集各后端的名称和特性信息
 */
const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

