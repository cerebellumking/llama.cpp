#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <android/log.h>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <jni.h>
#include <iomanip>
#include <limits>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <openssl/sha.h>
#include "llama.h"
#include "common.h"
#include "ggml-backend.h"
#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_client.hpp"
#include "heterospec_hotpath.hpp"
#include <thread>
#include <chrono>
// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("llama-android");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {
//         System.loadLibrary("llama-android")
//      }
//    }

#define TAG "llama-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

jclass la_int_var;
jmethodID la_int_var_value;
jmethodID la_int_var_inc;

static bool ensure_int_var_jni(JNIEnv *env, jobject intvar) {
    if (la_int_var_value != nullptr && la_int_var_inc != nullptr) {
        return true;
    }
    if (intvar == nullptr) {
        LOGe("IntVar jobject is null");
        return false;
    }
    jclass local = env->GetObjectClass(intvar);
    if (local == nullptr) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        LOGe("IntVar GetObjectClass failed");
        return false;
    }
    if (la_int_var == nullptr) {
        la_int_var = reinterpret_cast<jclass>(env->NewGlobalRef(local));
    }
    if (la_int_var_value == nullptr) {
        la_int_var_value = env->GetMethodID(local, "getValue", "()I");
        if (la_int_var_value == nullptr) {
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
            LOGe("IntVar.getValue() JNI lookup failed");
            env->DeleteLocalRef(local);
            return false;
        }
    }
    if (la_int_var_inc == nullptr) {
        la_int_var_inc = env->GetMethodID(local, "inc", "()V");
        if (la_int_var_inc == nullptr) {
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
            LOGe("IntVar.inc() JNI lookup failed; R8 likely stripped the method");
            env->DeleteLocalRef(local);
            return false;
        }
    }
    env->DeleteLocalRef(local);
    return true;
}

std::string cached_token_chars;

bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const unsigned char * bytes = (const unsigned char *)string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static void log_callback(ggml_log_level level, const char * fmt, void * data) {
    if (level == GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1model(JNIEnv *env, jobject, jstring filename) {
    llama_model_params model_params = llama_model_default_params();
    // Kotlin rejects truncated GGUFs before this call. mmap + extra_bufts keeps
    // KleidiAI CPU_REPACK. Never load CLIP/mmproj on this path (text-only).
    model_params.use_mmap = true;
    model_params.use_extra_bufts = true;

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("Loading model from %s (mmap=1 extra_bufts=1 text-only)", path_to_model);

    auto model = llama_model_load_from_file(path_to_model, model_params);
    env->ReleaseStringUTFChars(filename, path_to_model);

    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
        return 0;
    }

    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1model(JNIEnv *, jobject, jlong model) {
    llama_model_free(reinterpret_cast<llama_model *>(model));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1context(JNIEnv *env, jobject, jlong jmodel) {
    auto model = reinterpret_cast<llama_model *>(jmodel);

    if (!model) {
        LOGe("new_context(): model cannot be null");
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model cannot be null");
        return 0;
    }

    char desc[160] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    const bool hybrid = llama_model_is_hybrid(model) || llama_model_is_recurrent(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    const int nextn = llama_model_n_layer_nextn(model);

    int nproc = (int) sysconf(_SC_NPROCESSORS_ONLN);
    int n_threads = heterospec_hotpath::decode_thread_count(nproc);
    const auto pinned = heterospec_hotpath::big_core_ids(nproc, n_threads);
    const bool pinned_ok = heterospec_hotpath::pin_to_cpus(pinned);
    LOGi("Using %d threads (nproc=%d pin=%d kleidiai=%d hybrid=%d nextn=%d text-only)",
            n_threads,
            nproc,
            pinned_ok ? 1 : 0,
#if defined(GGML_CPU_KLEIDIAI) && GGML_CPU_KLEIDIAI
            1
#else
            0
#endif
            ,
            hybrid ? 1 : 0,
            nextn);

    llama_context_params ctx_params = llama_context_default_params();

    // n_outputs_max defaults to n_batch. 1536 * Qwen3.5 vocab (248320) * 4B ≈ 1.5 GiB
    // just for logits, which LMK-kills the process when switching onto 3.5-0.8B.
    // Hybrid Qwen3.5: 6 threads on the biggest cores, n_ctx=3072, KleidiAI.
    // Flash-attn only touches the 6 full-attn layers; GDN is unchanged.
    ctx_params.n_ctx           = 3072;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_batch         = hybrid ? 512 : 1536;
    ctx_params.n_ubatch        = hybrid ? 128 : 512;
    ctx_params.n_outputs_max   = 8;
    ctx_params.embeddings      = false;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    // Hybrid recurrent state cannot be cropped in the middle (n_rs_seq == 0).
    // Seq 0 keeps the confirmed prefix; seq 1 is the draft / overlap branch.
    if (hybrid) {
        ctx_params.n_seq_max = 2;
        ctx_params.kv_unified = true;
    }

    LOGi("new_context desc=%s hybrid=%d n_vocab=%d n_ctx=%u n_batch=%u n_ubatch=%u n_outputs_max=%u n_seq_max=%u kv_unified=%d fa=auto",
            desc,
            hybrid ? 1 : 0,
            n_vocab,
            ctx_params.n_ctx,
            ctx_params.n_batch,
            ctx_params.n_ubatch,
            ctx_params.n_outputs_max,
            ctx_params.n_seq_max,
            ctx_params.kv_unified ? 1 : 0);

    auto try_init = [&](llama_context_params params) -> llama_context * {
        try {
            return llama_init_from_model(model, params);
        } catch (const std::exception & e) {
            LOGe("llama_init_from_model threw: %s", e.what());
            return nullptr;
        }
    };

    llama_context * context = try_init(ctx_params);
    if (!context) {
        LOGe("new_context auto flash-attn failed, retry disabled");
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context = try_init(ctx_params);
    }
    if (!context && hybrid && ctx_params.n_seq_max > 1) {
        LOGe("hetero_phone seq_branch_unavailable retry n_seq_max=1");
        ctx_params.n_seq_max = 1;
        ctx_params.kv_unified = false;
        context = try_init(ctx_params);
    }

    if (!context) {
        LOGe("llama_new_context_with_model() returned null desc=%s hybrid=%d n_vocab=%d",
                desc, hybrid ? 1 : 0, n_vocab);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
        return 0;
    }

    return reinterpret_cast<jlong>(context);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1context(JNIEnv *, jobject, jlong context) {
    llama_free(reinterpret_cast<llama_context *>(context));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1free(JNIEnv *, jobject) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_log_1to_1android(JNIEnv *, jobject) {
    llama_log_set(log_callback, NULL);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1batch(JNIEnv *, jobject, jint n_tokens, jint embd, jint n_seq_max) {

    // Source: Copy of llama.cpp:llama_batch_init but heap-allocated.

    llama_batch *batch = new llama_batch {
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch->pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch->n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    batch->seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch->logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);

    return reinterpret_cast<jlong>(batch);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1batch(JNIEnv *, jobject, jlong batch_pointer) {
    //llama_batch_free(*reinterpret_cast<llama_batch *>(batch_pointer));
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    delete batch;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1sampler(JNIEnv *, jobject, jboolean minicpm_chat) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (minicpm_chat == JNI_TRUE) {
        // OpenBMB MiniCPM5-1B No-think: temp=0.7, top_p=0.95, min_p=0.
        // Greedy (and llama.cpp's default min_p=0.05) can get stuck repeating `**`.
        LOGi("MiniCPM5 sampler: top_p=0.95 temp=0.7 dist");
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1sampler(JNIEnv *, jobject, jlong sampler_pointer) {
    llama_sampler_free(reinterpret_cast<llama_sampler *>(sampler_pointer));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1init(JNIEnv *env, jobject, jstring native_lib_dir) {
    if (native_lib_dir != nullptr) {
        const char *path_to_backend = env->GetStringUTFChars(native_lib_dir, 0);
        if (path_to_backend && path_to_backend[0] != '\0') {
            LOGi("Loading backends from %s", path_to_backend);
            ggml_backend_load_all_from_path(path_to_backend);
            LOGi("Requested backends from %s (KleidiAI CPU variants must be in this dir)", path_to_backend);
        } else {
            LOGi("nativeLibraryDir empty; calling ggml_backend_load_all()");
            ggml_backend_load_all();
        }
        env->ReleaseStringUTFChars(native_lib_dir, path_to_backend);
    } else {
        ggml_backend_load_all();
    }
    llama_backend_init();
    LOGi("Backend initiated; registry count=%zu", ggml_backend_reg_count());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_system_1info(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1init(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jstring jtext,
        jboolean format_chat,
        jint n_len
    ) {

    cached_token_chars.clear();

    const auto text = env->GetStringUTFChars(jtext, 0);
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    bool parse_special = (format_chat == JNI_TRUE);
    const auto tokens_list = common_tokenize(context, text, true, parse_special);

    auto n_ctx = llama_n_ctx(context);
    auto n_kv_req = tokens_list.size() + n_len;

    LOGi("n_len = %d, n_ctx = %d, n_kv_req = %zu", n_len, n_ctx, n_kv_req);

    if (n_kv_req > n_ctx) {
        LOGe("error: n_kv_req > n_ctx, the required KV cache size is not big enough");
    }

    for (auto id : tokens_list) {
        LOGi("token: `%s`-> %d ", common_token_to_piece(context, id).c_str(), id);
    }

    common_batch_clear(*batch);

    // evaluate the initial prompt
    for (auto i = 0; i < tokens_list.size(); i++) {
        common_batch_add(*batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch->logits[batch->n_tokens - 1] = true;

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() failed");
    }

    env->ReleaseStringUTFChars(jtext, text);

    return batch->n_tokens;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1loop(
        JNIEnv * env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint n_len,
        jobject intvar_ncur
) {
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch   = reinterpret_cast<llama_batch   *>(batch_pointer);
    const auto sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);
    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);

    if (!ensure_int_var_jni(env, intvar_ncur)) {
        return nullptr;
    }

    // sample the most likely token
    const auto new_token_id = llama_sampler_sample(sampler, context, -1);

    const auto n_cur = env->CallIntMethod(intvar_ncur, la_int_var_value);
    if (llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len) {
        return nullptr;
    }

    auto new_token_chars = common_token_to_piece(context, new_token_id);
    cached_token_chars += new_token_chars;

    jstring new_token = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        new_token = env->NewStringUTF(cached_token_chars.c_str());
        LOGi("cached: %s, new_token_chars: `%s`, id: %d", cached_token_chars.c_str(), new_token_chars.c_str(), new_token_id);
        cached_token_chars.clear();
    } else {
        new_token = env->NewStringUTF("");
    }

    common_batch_clear(*batch);
    common_batch_add(*batch, new_token_id, n_cur, { 0 }, true);

    env->CallVoidMethod(intvar_ncur, la_int_var_inc);

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() returned null");
    }

    return new_token;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_kv_1cache_1clear(JNIEnv *, jobject, jlong context) {
    llama_memory_clear(llama_get_memory(reinterpret_cast<llama_context *>(context)), true);
}

// HeteroSpec

typedef websocketpp::client<websocketpp::config::asio_client> ws_client;

namespace heterospec_v3 {
constexpr uint8_t MSG_PREFILL_REQUEST = 0x10;
constexpr uint8_t MSG_VERIFY_REQUEST  = 0x11;
constexpr uint8_t MSG_CLOSE           = 0x12;
constexpr uint8_t MSG_PREFILL_ACK     = 0x90;
constexpr uint8_t MSG_VERIFY_ACK      = 0x91;
constexpr uint8_t MSG_ERROR           = 0x7F;
}

static void append_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 16u) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 24u) & 0xFFu));
}

static void append_i32_le(std::vector<uint8_t>& out, int32_t value) {
    append_u32_le(out, static_cast<uint32_t>(value));
}

static bool read_u32_le(const uint8_t * data, size_t size, size_t offset, uint32_t& value) {
    if (offset + sizeof(uint32_t) > size) {
        return false;
    }
    value = static_cast<uint32_t>(data[offset])
            | (static_cast<uint32_t>(data[offset + 1]) << 8u)
            | (static_cast<uint32_t>(data[offset + 2]) << 16u)
            | (static_cast<uint32_t>(data[offset + 3]) << 24u);
    return true;
}

static bool read_i32_le(const uint8_t * data, size_t size, size_t offset, int32_t& value) {
    uint32_t raw = 0;
    if (!read_u32_le(data, size, offset, raw)) {
        return false;
    }
    value = static_cast<int32_t>(raw);
    return true;
}

static std::vector<uint8_t> pack_token_request_v3(
        uint8_t type,
        uint32_t request_id,
        const llama_tokens& tokens) {
    std::vector<uint8_t> out;
    out.reserve(1 + 2 * sizeof(uint32_t) + tokens.size() * sizeof(int32_t));
    out.push_back(type);
    append_u32_le(out, request_id);
    append_u32_le(out, static_cast<uint32_t>(tokens.size()));
    for (llama_token token : tokens) {
        append_i32_le(out, static_cast<int32_t>(token));
    }
    return out;
}

static std::vector<uint8_t> pack_close_request_v3(uint32_t request_id) {
    std::vector<uint8_t> out;
    out.reserve(1 + sizeof(uint32_t));
    out.push_back(heterospec_v3::MSG_CLOSE);
    append_u32_le(out, request_id);
    return out;
}

enum class DraftStrategy {
    Fixed3,
    Adaptive,
};

static DraftStrategy parse_draft_strategy(const char * value) {
    if (value != nullptr && std::strcmp(value, "adaptive") == 0) {
        return DraftStrategy::Adaptive;
    }
    return DraftStrategy::Fixed3;
}

static const char * draft_strategy_name(DraftStrategy strategy) {
    return strategy == DraftStrategy::Adaptive ? "adaptive" : "fixed3";
}

struct CloudState {
    ws_client client;
    websocketpp::connection_hdl connection;
    std::string server_url;
    bool is_connected = false;
    std::vector<int> accepted_tokens;
    uint32_t next_request_id = 1;
    uint32_t pending_request_id = 0;
    uint32_t response_request_id = 0;
    uint32_t response_verify_us = 0;
    uint32_t response_queue_us = 0;
    uint32_t response_flags = 0;
    int response_accepted_draft_count = 0;
    bool has_prefill_ack = false;
    bool has_verify_response = false;
    bool has_error = false;
    uint32_t error_request_id = 0;
    uint32_t error_code = 0;
    std::string error_message;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<llama_token> prompt_tokens;
    int n_past{};
    llama_token last_token{};
    std::thread client_thread;
    std::condition_variable connect_cv;
    bool connection_ready = false;

    // 添加draft time存储数组
    std::vector<double> draft_times;

    DraftStrategy draft_strategy = DraftStrategy::Fixed3;
    std::string target_model = "qwen3-4b";
    std::string tokenizer_id = "qwen3-4b-tokenizer";
    bool use_queue_in_goodput = false;
    bool exclude_next_round_as_ttft = true;
    int adaptive_n_draft = 3;
    int min_n_draft = 1;
    int max_n_draft = 6;
    static constexpr int kFixedNDraft = 3;
    double draft_token_time_ema = 0.0;
    double verify_wait_time_ema = 0.0;
    double accept_rate_ema = 0.5;
    double ema_alpha = 0.3;
    double slo_tpot = 0.120;
    double default_draft_per_token = 0.020;
    double default_verify_wait_time = 0.080;
    double default_rtt = 0.040;
    double default_uplink_bytes_per_second = 4000.0;
    double rtt_ema = 0.0;
    double uplink_bytes_per_second_ema = 0.0;
    double queue_time_ema = 0.0;
    bool draft_token_time_ema_ready = false;
    bool verify_wait_time_ema_ready = false;
    bool rtt_ema_ready = false;
    bool uplink_bytes_per_second_ema_ready = false;
    bool queue_time_ema_ready = false;
    std::array<double, 11> confidence_bucket_accept_ema{};
    std::array<bool, 11> confidence_bucket_ready{};
    std::array<double, 6> position_accept_ema{};
    std::array<bool, 6> position_accept_ready{};
    int64_t icn_multi_total_wasted_tokens = 0;
    int64_t icn_multi_total_payload_bytes = 0;
    bool has_draft_batch = false;
    llama_batch draft_batch{};
    bool has_prefetch = false;
    llama_tokens prefetch_drafts;
    std::vector<float> prefetch_confidences;
    std::vector<float> prefetch_calibrated;
    // Hybrid/recurrent models keep the confirmed prefix on seq 0 and draft on seq 1.
    bool branch_kv = false;
    bool branch_ready = false;
};

static uint32_t cloud_next_request_id(CloudState& state) {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.next_request_id == 0) {
        state.next_request_id = 1;
    }
    return state.next_request_id++;
}

static void cloud_prepare_pending(CloudState& state, uint32_t request_id) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.pending_request_id = request_id;
    state.response_request_id = 0;
    state.response_verify_us = 0;
    state.response_queue_us = 0;
    state.response_flags = 0;
    state.response_accepted_draft_count = 0;
    state.accepted_tokens.clear();
    state.has_prefill_ack = false;
    state.has_verify_response = false;
    state.has_error = false;
    state.error_request_id = 0;
    state.error_code = 0;
    state.error_message.clear();
}

static bool parse_v3_verify_ack(
        const uint8_t * data,
        size_t size,
        uint32_t& request_id,
        int& accepted_draft_count,
        uint32_t& verify_us,
        uint32_t& queue_us,
        uint32_t& flags,
        std::vector<int>& tokens) {
    constexpr size_t header_size = 1 + 6 * sizeof(uint32_t);
    if (size < header_size) {
        return false;
    }

    uint32_t token_count = 0;
    uint32_t accepted_count_u32 = 0;
    if (!read_u32_le(data, size, 1, request_id)
            || !read_u32_le(data, size, 5, accepted_count_u32)
            || !read_u32_le(data, size, 9, token_count)
            || !read_u32_le(data, size, 13, verify_us)
            || !read_u32_le(data, size, 17, queue_us)
            || !read_u32_le(data, size, 21, flags)) {
        return false;
    }
    accepted_draft_count = static_cast<int>(accepted_count_u32);
    if (size != header_size + static_cast<size_t>(token_count) * sizeof(int32_t)) {
        return false;
    }

    tokens.clear();
    tokens.reserve(token_count);
    for (uint32_t i = 0; i < token_count; ++i) {
        int32_t token = 0;
        if (!read_i32_le(data, size, header_size + static_cast<size_t>(i) * sizeof(int32_t), token)) {
            return false;
        }
        tokens.push_back(static_cast<int>(token));
    }
    return true;
}

static bool parse_v3_prefill_ack(
        const uint8_t * data,
        size_t size,
        uint32_t& request_id) {
    constexpr size_t header_size = 1 + 2 * sizeof(uint32_t);
    uint32_t metadata_len = 0;
    return size >= header_size
            && read_u32_le(data, size, 1, request_id)
            && read_u32_le(data, size, 5, metadata_len)
            && size == header_size + metadata_len;
}

static bool parse_v3_error(
        const uint8_t * data,
        size_t size,
        uint32_t& request_id,
        uint32_t& code,
        std::string& message) {
    constexpr size_t header_size = 1 + 3 * sizeof(uint32_t);
    uint32_t message_len = 0;
    if (size < header_size
            || !read_u32_le(data, size, 1, request_id)
            || !read_u32_le(data, size, 5, code)
            || !read_u32_le(data, size, 9, message_len)
            || size != header_size + message_len) {
        return false;
    }

    message.assign(
            reinterpret_cast<const char *>(data + header_size),
            reinterpret_cast<const char *>(data + header_size + message_len));
    return true;
}

static double update_ema_value(double previous, bool& ready, double value, double alpha) {
    if (!ready) {
        ready = true;
        return value;
    }
    return alpha * value + (1.0 - alpha) * previous;
}

static int clamp_adaptive_n_draft(const CloudState& state, int value) {
    return std::max(state.min_n_draft, std::min(state.max_n_draft, value));
}

static int icn_multi_confidence_bucket(double confidence) {
    const double bounded = std::max(0.0, std::min(1.0, confidence));
    return std::max(0, std::min(10, static_cast<int>(bounded / 0.1)));
}

static double icn_multi_calibrate_confidence_value(const CloudState& state, float confidence, int position) {
    const double raw = std::max(0.0, std::min(1.0, static_cast<double>(confidence)));
    const int bucket = icn_multi_confidence_bucket(raw);
    const double bucket_rate = state.confidence_bucket_ready[bucket]
            ? state.confidence_bucket_accept_ema[bucket]
            : raw;
    const bool position_ready = position >= 0
            && position < static_cast<int>(state.position_accept_ready.size())
            && state.position_accept_ready[position];
    const double position_rate = position_ready
            ? state.position_accept_ema[position]
            : raw;
    return std::max(0.0, std::min(1.0, 0.6 * bucket_rate + 0.3 * position_rate + 0.1 * raw));
}

static void icn_multi_update_calibrator(
        CloudState& state,
        const std::vector<float>& raw_confidences,
        int accepted_draft_count) {
    const int bounded_accepted = std::max(0, std::min(accepted_draft_count, static_cast<int>(raw_confidences.size())));
    for (int i = 0; i < static_cast<int>(raw_confidences.size()); ++i) {
        const double accepted = i < bounded_accepted ? 1.0 : 0.0;
        const int bucket = icn_multi_confidence_bucket(raw_confidences[i]);
        state.confidence_bucket_accept_ema[bucket] = update_ema_value(
                state.confidence_bucket_accept_ema[bucket],
                state.confidence_bucket_ready[bucket],
                accepted,
                state.ema_alpha);
        if (i < static_cast<int>(state.position_accept_ema.size())) {
            state.position_accept_ema[i] = update_ema_value(
                    state.position_accept_ema[i],
                    state.position_accept_ready[i],
                    accepted,
                    state.ema_alpha);
        }
    }
}

static std::string icn_multi_sha256_first16_hex(const std::string& payload) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(payload.data()), payload.size(), hash);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        out << std::setw(2) << static_cast<int>(hash[i]);
    }
    return out.str();
}

static std::string icn_multi_json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out << '\\';
        }
        out << c;
    }
    out << '"';
    return out.str();
}

static std::string icn_multi_hash_tokens(const llama_tokens& tokens) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << tokens[i];
    }
    out << ']';
    return icn_multi_sha256_first16_hex(out.str());
}

static std::string icn_multi_url_quote(const std::string& value) {
    std::ostringstream out;
    out << std::uppercase << std::hex << std::setfill('0');
    for (unsigned char c : value) {
        const bool safe = (c >= 'a' && c <= 'z')
                || (c >= 'A' && c <= 'Z')
                || (c >= '0' && c <= '9')
                || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out << static_cast<char>(c);
        } else {
            out << '%' << std::setw(2) << static_cast<int>(c);
        }
    }
    return out.str();
}

static std::string jstring_to_utf8(JNIEnv *env, jstring value) {
    if (value == nullptr) {
        return "";
    }
    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return "";
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

static std::string icn_multi_build_icn_name_v2(
        const CloudState& state,
        const llama_tokens& prefix_tokens,
        const llama_tokens& draft_tokens,
        int speculative_len) {
    const std::string target_model = state.target_model.empty() ? "qwen3-4b" : state.target_model;
    const std::string draft_model = "android-local-draft";
    const std::string tokenizer_id = state.tokenizer_id.empty()
            ? (target_model + "-tokenizer")
            : state.tokenizer_id;
    const std::string sampling_config = "{\"temperature\":0,\"top_k\":1}";
    const std::string policy = "observe_only";

    return "/6g-ai/heterospec/v2/"
            + icn_multi_url_quote(target_model)
            + "/"
            + icn_multi_url_quote(draft_model)
            + "/tok="
            + icn_multi_sha256_first16_hex(icn_multi_json_string(tokenizer_id))
            + "/sampler="
            + icn_multi_sha256_first16_hex(sampling_config)
            + "/policy="
            + icn_multi_url_quote(policy)
            + "/prefix="
            + icn_multi_hash_tokens(prefix_tokens)
            + "/draft="
            + icn_multi_hash_tokens(draft_tokens)
            + "/sl="
            + std::to_string(speculative_len);
}

static std::string join_token_ids(const llama_tokens& tokens) {
    std::ostringstream out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << static_cast<int32_t>(tokens[i]);
    }
    return out.str();
}

static std::string icn_multi_join_floats(const std::vector<float>& values) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    return out.str();
}

static double adaspec_expected_tokens(const std::vector<float>& confidences) {
    double total = 1.0;
    double prefix = 1.0;
    for (float confidence : confidences) {
        const double bounded = std::max(0.0, std::min(1.0, static_cast<double>(confidence)));
        prefix *= bounded;
        total += prefix;
    }
    return total;
}

// queue_time must be 0 unless CloudState.use_queue_in_goodput is on.
// Current queue_us is bonus-precompute wait on this connection, not shared GPU queue.
static double icn_multi_goodput(
        const std::vector<float>& confidences,
        double draft_per_token,
        double verify_wait_time,
        double network_time,
        double queue_time,
        double slo_tpot) {
    if (confidences.empty()) {
        return -1.0;
    }

    const double draft_time = std::max(0.0, draft_per_token) * confidences.size();
    const double total_time = draft_time
            + std::max(0.0, verify_wait_time)
            + std::max(0.0, network_time)
            + std::max(0.0, queue_time);
    if (total_time <= 0.0) {
        return -1.0;
    }

    const double expected = adaspec_expected_tokens(confidences);
    const double estimated_tpot = total_time / expected;
    if (slo_tpot > 0.0 && estimated_tpot > slo_tpot) {
        return -1.0;
    }

    return expected / total_time;
}

struct AdaptiveDraftResult {
    llama_tokens tokens;
    std::vector<float> confidences;
    std::vector<float> calibrated_confidences;
};

static bool icn_multi_select_greedy_token(
        llama_context * context,
        const llama_vocab * vocab,
        llama_token& token,
        float& confidence) {
    const float * logits = llama_get_logits_ith(context, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int32_t selected = 0;
    if (!heterospec_hotpath::greedy_argmax(logits, n_vocab, selected, confidence)) {
        return false;
    }
    token = static_cast<llama_token>(selected);
    return true;
}

static llama_batch& ensure_draft_batch(CloudState& state) {
    if (!state.has_draft_batch) {
        state.draft_batch = llama_batch_init(state.max_n_draft + 1, 0, 1);
        state.has_draft_batch = true;
    }
    return state.draft_batch;
}

static llama_seq_id draft_seq(const CloudState& state) {
    return state.branch_kv ? static_cast<llama_seq_id>(1) : static_cast<llama_seq_id>(0);
}

static AdaptiveDraftResult generate_icn_multi_local_draft(
        CloudState& state,
        llama_context * context,
        const llama_vocab * vocab,
        const llama_tokens& prompt_tgt,
        llama_token id_last,
        double draft_per_token,
        double verify_wait_time,
        double network_time,
        double network_time_per_token,
        double queue_time) {
    AdaptiveDraftResult result;
    const bool adaptive = state.draft_strategy == DraftStrategy::Adaptive;
    const int n_draft_max = clamp_adaptive_n_draft(
            state,
            adaptive ? state.max_n_draft : CloudState::kFixedNDraft);
    const double queue_time_for_goodput = state.use_queue_in_goodput
            ? std::max(0.0, queue_time)
            : 0.0;
    result.tokens.reserve(n_draft_max);
    result.confidences.reserve(n_draft_max);
    result.calibrated_confidences.reserve(n_draft_max);

    llama_batch& batch = ensure_draft_batch(state);
    const llama_pos n_past = static_cast<llama_pos>(prompt_tgt.size());

    common_batch_clear(batch);
    common_batch_add(batch, id_last, n_past, { draft_seq(state) }, true);
    if (llama_decode(context, batch) != 0) {
        LOGe("hetero_phone draft_sync_fail last=%d n_past=%d",
                static_cast<int>(id_last),
                static_cast<int>(n_past));
        return result;
    }

    double best_goodput = -std::numeric_limits<double>::infinity();
    bool found_feasible = false;

    for (int i = 0; i < n_draft_max; ++i) {
        llama_token id = 0;
        float confidence = 0.0f;
        if (!icn_multi_select_greedy_token(context, vocab, id, confidence)) {
            LOGe("ICN-Multi-HeteroSpec draft failed to select token at position=%d", i);
            break;
        }

        const float calibrated = static_cast<float>(
                icn_multi_calibrate_confidence_value(state, confidence, i));

        result.tokens.push_back(id);
        result.confidences.push_back(confidence);
        result.calibrated_confidences.push_back(calibrated);

        if (adaptive) {
            const double candidate_network_time =
                    std::max(0.0, network_time)
                    + std::max(0.0, network_time_per_token) * result.tokens.size();
            const double goodput = icn_multi_goodput(
                    result.calibrated_confidences,
                    draft_per_token,
                    verify_wait_time,
                    candidate_network_time,
                    queue_time_for_goodput,
                    state.slo_tpot);

            if (goodput > best_goodput) {
                best_goodput = goodput;
                found_feasible = goodput >= 0.0;
                if (!found_feasible) {
                    break;
                }
            } else if (found_feasible) {
                result.tokens.pop_back();
                result.confidences.pop_back();
                result.calibrated_confidences.pop_back();
                break;
            }
        }

        if (n_draft_max <= static_cast<int>(result.tokens.size())) {
            break;
        }

        common_batch_clear(batch);
        common_batch_add(batch, id, n_past + i + 1, { draft_seq(state) }, true);
        if (llama_decode(context, batch) != 0) {
            LOGe("hetero_phone draft_decode_fail position=%d", i);
            break;
        }
    }

    return result;
}

static bool verify_ack_ready(CloudState& state) {
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.has_verify_response || state.has_error || !state.is_connected;
}

struct OverlapPrefetch {
    bool decoded_last_sent = false;
    bool has_guessed_bonus = false;
    llama_token guessed_bonus = 0;
    llama_tokens drafts;
    std::vector<float> confidences;
    std::vector<float> calibrated;
};

static void overlap_while_waiting(
        CloudState& state,
        llama_context * context,
        const llama_vocab * vocab,
        const llama_tokens& sent_drafts,
        llama_pos pos_base,
        int n_draft_max,
        OverlapPrefetch& prefetch) {
    if (sent_drafts.empty() || n_draft_max <= 0 || verify_ack_ready(state)) {
        return;
    }
    llama_batch& batch = ensure_draft_batch(state);
    common_batch_clear(batch);
    common_batch_add(
            batch,
            sent_drafts.back(),
            pos_base + static_cast<llama_pos>(sent_drafts.size()),
            { draft_seq(state) },
            true);
    if (llama_decode(context, batch) != 0) {
        return;
    }
    prefetch.decoded_last_sent = true;
    while (!verify_ack_ready(state) && static_cast<int>(prefetch.drafts.size()) < n_draft_max) {
        llama_token id = 0;
        float confidence = 0.0f;
        if (!icn_multi_select_greedy_token(context, vocab, id, confidence)) {
            break;
        }
        const float calibrated = static_cast<float>(
                icn_multi_calibrate_confidence_value(
                        state,
                        confidence,
                        prefetch.has_guessed_bonus
                                ? static_cast<int>(prefetch.drafts.size())
                                : 0));
        if (!prefetch.has_guessed_bonus) {
            prefetch.guessed_bonus = id;
            prefetch.has_guessed_bonus = true;
            common_batch_clear(batch);
            common_batch_add(
                    batch,
                    id,
                    pos_base + static_cast<llama_pos>(sent_drafts.size()) + 1,
                    { draft_seq(state) },
                    true);
            if (llama_decode(context, batch) != 0) {
                prefetch.has_guessed_bonus = false;
                break;
            }
            continue;
        }
        prefetch.drafts.push_back(id);
        prefetch.confidences.push_back(confidence);
        prefetch.calibrated.push_back(calibrated);
        if (static_cast<int>(prefetch.drafts.size()) >= n_draft_max) {
            break;
        }
        common_batch_clear(batch);
        common_batch_add(
                batch,
                id,
                pos_base + static_cast<llama_pos>(sent_drafts.size()) + 1
                        + static_cast<llama_pos>(prefetch.drafts.size()),
                { draft_seq(state) },
                true);
        if (llama_decode(context, batch) != 0) {
            prefetch.drafts.pop_back();
            prefetch.confidences.pop_back();
            prefetch.calibrated.pop_back();
            break;
        }
    }
}

// Write [previous last token] + accepted drafts into seq 0. The bonus is the
// next round's last token and stays out of this prefix. Logits are requested
// only for the last committed token, same as prefill.
static bool commit_to_seq0(
        llama_context * context,
        CloudState& state,
        llama_token previous_last,
        const llama_tokens& drafts,
        int accepted,
        int pos0) {
    const int n_accepted = std::max(0, std::min(accepted, static_cast<int>(drafts.size())));
    const int n_commit = heterospec_hotpath::commit_token_count(n_accepted);
    if (n_commit > state.max_n_draft + 1) {
        LOGe("hetero_phone commit_fail pos0=%d n=%d cap=%d",
                pos0,
                n_commit,
                state.max_n_draft + 1);
        return false;
    }
    llama_batch& batch = ensure_draft_batch(state);
    common_batch_clear(batch);
    common_batch_add(batch, previous_last, pos0, { 0 }, n_commit == 1);
    for (int i = 0; i < n_accepted; ++i) {
        common_batch_add(
                batch,
                drafts[static_cast<size_t>(i)],
                pos0 + 1 + i,
                { 0 },
                i + 1 == n_accepted);
    }
    if (llama_decode(context, batch) != 0) {
        LOGe("hetero_phone commit_fail pos0=%d n=%d", pos0, n_commit);
        return false;
    }
    LOGi("hetero_phone commit pos0=%d n=%d", pos0, n_commit);
    return true;
}

static bool prefill_prefix(llama_context * context, const llama_tokens& tokens) {
    if (tokens.empty()) {
        return false;
    }
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(context)));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    bool ok = true;
    for (size_t start = 0; start < tokens.size() && ok; ) {
        common_batch_clear(batch);
        const size_t end = std::min(tokens.size(), start + static_cast<size_t>(n_batch));
        for (size_t i = start; i < end; ++i) {
            common_batch_add(
                    batch,
                    tokens[i],
                    static_cast<llama_pos>(i),
                    { 0 },
                    i + 1 == tokens.size());
        }
        if (llama_decode(context, batch) != 0) {
            ok = false;
        }
        start = end;
    }
    llama_batch_free(batch);
    return ok;
}

static void update_adaptive_n_draft(
        CloudState* state,
        double draft_time,
        int draft_count,
        double verify_wait_time,
        int accepted_draft_count,
        const std::vector<float>& confidence_scores,
        const std::vector<float>& calibrated_confidence_scores,
        double network_time,
        double queue_time,
        int payload_bytes) {
    if (!state || draft_count <= 0) {
        return;
    }

    const double draft_per_token = std::max(0.0, draft_time) / draft_count;
    if (!std::isfinite(draft_per_token) || draft_per_token <= 0.0) {
        return;
    }

    const double wait_time = std::max(0.0, verify_wait_time);
    const int bounded_accepted = std::max(0, std::min(accepted_draft_count, draft_count));
    const double accept_rate = static_cast<double>(bounded_accepted) / draft_count;

    state->draft_token_time_ema = update_ema_value(
            state->draft_token_time_ema,
            state->draft_token_time_ema_ready,
            draft_per_token,
            state->ema_alpha);
    state->verify_wait_time_ema = update_ema_value(
            state->verify_wait_time_ema,
            state->verify_wait_time_ema_ready,
            wait_time,
            state->ema_alpha);
    state->accept_rate_ema =
            state->ema_alpha * accept_rate + (1.0 - state->ema_alpha) * state->accept_rate_ema;
    state->rtt_ema = update_ema_value(
            state->rtt_ema,
            state->rtt_ema_ready,
            std::max(0.001, std::min(0.250, network_time)),
            state->ema_alpha);
    const double observed_queue_time = std::max(0.0, queue_time);
    const double queue_time_for_goodput = state->use_queue_in_goodput ? observed_queue_time : 0.0;
    state->queue_time_ema = update_ema_value(
            state->queue_time_ema,
            state->queue_time_ema_ready,
            observed_queue_time,
            state->ema_alpha);
    state->icn_multi_total_wasted_tokens += std::max(0, draft_count - bounded_accepted);
    state->icn_multi_total_payload_bytes += std::max(0, payload_bytes);
    icn_multi_update_calibrator(*state, confidence_scores, bounded_accepted);

    state->adaptive_n_draft = clamp_adaptive_n_draft(*state, draft_count);

    const double expected_tokens = adaspec_expected_tokens(calibrated_confidence_scores.empty()
            ? confidence_scores
            : calibrated_confidence_scores);
    const double total_time = std::max(0.0, draft_time)
            + wait_time
            + std::max(0.0, network_time)
            + queue_time_for_goodput;
    const double estimated_tpot = expected_tokens > 0.0 ? total_time / expected_tokens : total_time;
    const double goodput = total_time > 0.0 ? expected_tokens / total_time : 0.0;
    const double last_confidence = confidence_scores.empty() ? 0.0 : confidence_scores.back();
    const double last_calibrated = calibrated_confidence_scores.empty() ? last_confidence : calibrated_confidence_scores.back();

    LOGi("icn_multi_k=%d icn_multi_goodput=%.3f icn_multi_estimated_tpot=%.3fs icn_multi_raw_conf=%.3f icn_multi_calibrated_p=%.3f icn_multi_expected_tokens=%.3f icn_multi_network_time=%.3fs icn_multi_queue_time=%.3fs icn_multi_payload_bytes=%d icn_multi_wasted_tokens=%lld accept_rate=%.3f draft_time=%.3fs verify_wait_time=%.3fs",
            state->adaptive_n_draft,
            goodput,
            estimated_tpot,
            last_confidence,
            last_calibrated,
            expected_tokens,
            network_time,
            observed_queue_time,
            payload_bytes,
            static_cast<long long>(state->icn_multi_total_wasted_tokens),
            accept_rate,
            draft_time,
            verify_wait_time);
}

bool init_websocket(CloudState& state, const std::string& url) {
    if (state.is_connected) return true;

    state.client.clear_access_channels(websocketpp::log::alevel::all);
    state.client.set_access_channels(
            websocketpp::log::alevel::connect |
            websocketpp::log::alevel::disconnect |
            websocketpp::log::alevel::fail
    );

    state.client.clear_error_channels(websocketpp::log::elevel::all);
    state.client.set_error_channels(
            websocketpp::log::elevel::rerror |
            websocketpp::log::elevel::fatal
    );

    state.client.set_max_message_size(1024 * 1024);
    state.client.set_max_http_body_size(1024 * 1024);
    state.client.init_asio();
    state.client.start_perpetual();

    state.client.set_message_handler([&state](websocketpp::connection_hdl hdl, ws_client::message_ptr msg) {
        (void) hdl;
        try {
            const auto* data = reinterpret_cast<const uint8_t*>(msg->get_payload().data());
            const size_t size = msg->get_payload().size();
            if (size == 0) {
                return;
            }

            if (data[0] == heterospec_v3::MSG_PREFILL_ACK) {
                uint32_t request_id = 0;
                if (!parse_v3_prefill_ack(data, size, request_id)) {
                    LOGe("Malformed v3 prefill ack");
                    return;
                }
                std::unique_lock<std::mutex> lock(state.mutex);
                if (request_id != state.pending_request_id) {
                    LOGe("Ignoring stale prefill ack request_id=%u pending=%u",
                            request_id,
                            state.pending_request_id);
                    return;
                }
                state.response_request_id = request_id;
                state.has_prefill_ack = true;
                state.pending_request_id = 0;
                state.cv.notify_one();
                return;
            }

            if (data[0] == heterospec_v3::MSG_VERIFY_ACK) {
                std::vector<int> tokens;
                uint32_t request_id = 0;
                int accepted_draft_count = 0;
                uint32_t verify_us = 0;
                uint32_t queue_us = 0;
                uint32_t flags = 0;
                if (!parse_v3_verify_ack(
                            data,
                            size,
                            request_id,
                            accepted_draft_count,
                            verify_us,
                            queue_us,
                            flags,
                            tokens)) {
                    LOGe("Malformed v3 verify ack");
                    return;
                }

                std::unique_lock<std::mutex> lock(state.mutex);
                if (request_id != state.pending_request_id) {
                    LOGe("Ignoring stale verify ack request_id=%u pending=%u",
                            request_id,
                            state.pending_request_id);
                    return;
                }
                state.accepted_tokens = std::move(tokens);
                state.response_request_id = request_id;
                state.response_accepted_draft_count = accepted_draft_count;
                state.response_verify_us = verify_us;
                state.response_queue_us = queue_us;
                state.response_flags = flags;
                state.has_verify_response = true;
                state.pending_request_id = 0;
                state.cv.notify_one();
                return;
            }

            if (data[0] == heterospec_v3::MSG_ERROR) {
                uint32_t request_id = 0;
                uint32_t code = 0;
                std::string message;
                if (!parse_v3_error(data, size, request_id, code, message)) {
                    LOGe("Malformed v3 error message");
                    return;
                }

                std::unique_lock<std::mutex> lock(state.mutex);
                if (state.pending_request_id != 0 && request_id != state.pending_request_id) {
                    LOGe("Ignoring stale error request_id=%u pending=%u",
                            request_id,
                            state.pending_request_id);
                    return;
                }
                state.error_request_id = request_id;
                state.error_code = code;
                state.error_message = std::move(message);
                state.has_error = true;
                state.pending_request_id = 0;
                state.cv.notify_one();
            }
        } catch (const std::exception& e) {
            LOGe("Error parsing message: %s", e.what());
        }
    });

    state.client.set_open_handler([&state](websocketpp::connection_hdl hdl) {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.connection = hdl;
        state.is_connected = true;
        state.connection_ready = true;
        state.connect_cv.notify_all();
        LOGi("WebSocket connection opened");
    });

    state.client.set_fail_handler([&state](websocketpp::connection_hdl hdl) {
        (void) hdl;
        std::unique_lock<std::mutex> lock(state.mutex);
        state.is_connected = false;
        state.connection_ready = true;
        state.connect_cv.notify_all();
        state.cv.notify_all();
        LOGe("WebSocket connection failed");
    });

    state.client.set_close_handler([&state](websocketpp::connection_hdl hdl) {
        (void) hdl;
        std::unique_lock<std::mutex> lock(state.mutex);
        state.is_connected = false;
        state.cv.notify_all();
        LOGi("WebSocket connection closed");
    });

    websocketpp::lib::error_code ec;
    auto con = state.client.get_connection(url, ec);
    if (ec) {
        LOGe("Failed to create connection: %s", ec.message().c_str());
        state.client.stop_perpetual();
        return false;
    }

    state.client.connect(con);

    state.client_thread = std::thread([&state]() {
        try {
            state.client.run();
        } catch (const std::exception& e) {
            LOGe("WebSocket client thread error: %s", e.what());
        }
    });

    {
        std::unique_lock<std::mutex> lock(state.mutex);
        if (!state.connect_cv.wait_for(lock, std::chrono::seconds(15), [&state] {
            return state.connection_ready;
        })) {
            LOGe("WebSocket connection timeout");
            lock.unlock();
            state.client.stop_perpetual();
            state.client.stop();
            if (state.client_thread.joinable()) {
                state.client_thread.join();
            }
            return false;
        }
        if (!state.is_connected) {
            LOGe("WebSocket connection did not open");
            lock.unlock();
            state.client.stop_perpetual();
            state.client.stop();
            if (state.client_thread.joinable()) {
                state.client_thread.join();
            }
            return false;
        }
    }

    return true;
}

static void shutdown_cloud_state(CloudState& state) {
    if (state.is_connected) {
        try {
            const uint32_t request_id = cloud_next_request_id(state);
            const auto close_msg = pack_close_request_v3(request_id);
            state.client.send(
                    state.connection,
                    close_msg.data(),
                    close_msg.size(),
                    websocketpp::frame::opcode::binary);
        } catch (const std::exception& e) {
            LOGe("Exception while sending close message: %s", e.what());
        }

        try {
            websocketpp::lib::error_code ec;
            state.client.close(
                    state.connection,
                    websocketpp::close::status::normal,
                    "heterospec cleanup",
                    ec);
            if (ec) {
                LOGe("Exception while closing websocket: %s", ec.message().c_str());
            }
        } catch (const std::exception& e) {
            LOGe("Exception while closing websocket: %s", e.what());
        }
    }

    state.client.stop_perpetual();
    state.client.stop();
    if (state.client_thread.joinable()) {
        state.client_thread.join();
    }
    if (state.has_draft_batch) {
        llama_batch_free(state.draft_batch);
        state.has_draft_batch = false;
    }
    state.is_connected = false;
    state.connection_ready = false;
}

// 添加清理函数
extern "C" JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_heterospec_1cleanup(JNIEnv *env, jobject thiz) {
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeStatePtr", "J");
    auto* state = reinterpret_cast<CloudState*>(env->GetLongField(thiz, fid));

    if (state) {
        shutdown_cloud_state(*state);
        delete state;
        env->SetLongField(thiz, fid, 0);
    }
}

// 核心函数1: 初始化推测解码
extern "C" JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_heterospec_1init(
        JNIEnv *env,
        jobject thiz,
        jlong context_pointer,
        jlong batch_pointer,
        jstring jtext,
        jboolean format_chat,
        jint n_len,
        jstring server_url,
        jstring draft_strategy,
        jboolean use_queue_in_goodput,
        jstring target_model,
        jstring tokenizer_id)
{
    cached_token_chars.clear();
    const auto text = env->GetStringUTFChars(jtext, 0);
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    bool parse_special = (format_chat == JNI_TRUE);
    const auto tokens_list = common_tokenize(context, text, true, parse_special);
    if (tokens_list.size() < 2) {
        LOGe("heterospec_init requires at least two prompt tokens");
        env->ReleaseStringUTFChars(jtext, text);
        return -2;
    }

    const char *url = env->GetStringUTFChars(server_url, nullptr);
    const char *strategy_cstr = draft_strategy != nullptr
            ? env->GetStringUTFChars(draft_strategy, nullptr)
            : nullptr;
    auto state_owner = std::make_unique<CloudState>();
    auto* state = state_owner.get();
    state->server_url = url ? url : "";
    state->draft_strategy = parse_draft_strategy(strategy_cstr);
    state->use_queue_in_goodput = (use_queue_in_goodput == JNI_TRUE);
    state->target_model = jstring_to_utf8(env, target_model);
    if (state->target_model.empty()) {
        state->target_model = "qwen3-4b";
    }
    state->tokenizer_id = jstring_to_utf8(env, tokenizer_id);
    if (state->tokenizer_id.empty()) {
        state->tokenizer_id = state->target_model + "-tokenizer";
    }
    LOGi("hetero_config draft_strategy=%s use_queue_in_goodput=%d target=%s tokenizer=%s",
            draft_strategy_name(state->draft_strategy),
            state->use_queue_in_goodput ? 1 : 0,
            state->target_model.c_str(),
            state->tokenizer_id.c_str());
    if (strategy_cstr != nullptr) {
        env->ReleaseStringUTFChars(draft_strategy, strategy_cstr);
        strategy_cstr = nullptr;
    }

    // 初始化WebSocket连接
    if (!init_websocket(*state, url)) {
        env->ReleaseStringUTFChars(server_url, url);
        env->ReleaseStringUTFChars(jtext, text);
        return -2;
    }

    // 保存 prompt tokens 和最后一个 token
    llama_tokens prompt_tgt(tokens_list.begin(), tokens_list.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(context));
    state->n_past = prompt_tgt.size() - 1;
    state->prompt_tokens = prompt_tgt;
    state->last_token = tokens_list.back();
    {
        const llama_model * model = llama_get_model(context);
        const bool hybrid = model != nullptr
                && (llama_model_is_hybrid(model) || llama_model_is_recurrent(model));
        state->branch_kv = hybrid && llama_n_seq_max(context) >= 2;
        state->branch_ready = false;
        LOGi("hetero_phone branch_kv=%d n_seq_max=%u",
                state->branch_kv ? 1 : 0,
                llama_n_seq_max(context));
    }

    // 发送prefill请求
    try {
        if (!state->is_connected || !state->connection_ready) {
            LOGe("WebSocket not connected or not ready");
            env->ReleaseStringUTFChars(server_url, url);
            env->ReleaseStringUTFChars(jtext, text);
            return -2;
        }

        const uint32_t request_id = cloud_next_request_id(*state);
        const auto binary_msg = pack_token_request_v3(
                heterospec_v3::MSG_PREFILL_REQUEST,
                request_id,
                tokens_list);
        cloud_prepare_pending(*state, request_id);

        state->client.send(state->connection, binary_msg.data(), binary_msg.size(), websocketpp::frame::opcode::binary);
        LOGi("Prefill v3 message sent successfully request_id=%u", request_id);
    } catch (const std::exception& e) {
        LOGe("Exception while sending message: %s", e.what());
        shutdown_cloud_state(*state);
        env->ReleaseStringUTFChars(server_url, url);
        env->ReleaseStringUTFChars(jtext, text);
        return -2;
    }

    //  prefill draft model
    common_batch_clear(*batch);

    // evaluate the initial prompt
    for (auto i = 0; i < prompt_tgt.size(); i++) {
        common_batch_add(*batch, prompt_tgt[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch->logits[batch->n_tokens - 1] = true;

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() failed");
    }

    {
        std::unique_lock<std::mutex> lock(state->mutex);
        const bool ready = state->cv.wait_for(lock, std::chrono::seconds(30), [state] {
            return state->has_prefill_ack || state->has_error;
        });
        if (!ready || state->has_error) {
            LOGe("Prefill v3 failed ready=%d error_code=%u error=%s",
                    ready ? 1 : 0,
                    state->error_code,
                    state->error_message.c_str());
            lock.unlock();
            shutdown_cloud_state(*state);
            env->ReleaseStringUTFChars(server_url, url);
            env->ReleaseStringUTFChars(jtext, text);
            return -2;
        }
    }

    // 将状态指针保存到Java对象中
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeStatePtr", "J");
    env->SetLongField(thiz, fid, reinterpret_cast<jlong>(state_owner.release()));

    env->ReleaseStringUTFChars(server_url, url);
    env->ReleaseStringUTFChars(jtext, text);

    return batch->n_tokens;
}

// 核心函数2: 协同解码循环
extern "C" JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_heterospec_1loop(
        JNIEnv *env, jobject thiz,
        jlong context_pointer, jint n_len,
        jobject intvar_ncur)
{
    // 从Java对象获取状态指针
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeStatePtr", "J");
    auto* state = reinterpret_cast<CloudState*>(env->GetLongField(thiz, fid));

    if (!ensure_int_var_jni(env, intvar_ncur)) {
        return env->NewStringUTF("");
    }

    if (!state || !state->is_connected) {
        LOGe("hetero_phone loop_skip state=%d connected=%d",
                state != nullptr ? 1 : 0,
                state != nullptr && state->is_connected ? 1 : 0);
        return nullptr;
    }

    // 1. 本地生成推测token (使用llama.cpp的API)
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);

    // 使用保存的状态
    const auto& prompt_tgt = state->prompt_tokens;
    const auto id_last = state->last_token;
    LOGi("hetero_phone loop_enter n_past=%d prompt=%zu last=%d prefetch=%d",
            state->n_past,
            prompt_tgt.size(),
            static_cast<int>(id_last),
            state->has_prefetch ? 1 : 0);

    // 生成草稿token
    auto start1 = std::chrono::high_resolution_clock::now();
    const double estimated_draft_per_token = state->draft_token_time_ema_ready
            ? state->draft_token_time_ema
            : state->default_draft_per_token;
    const double estimated_verify_wait_time = state->verify_wait_time_ema_ready
            ? state->verify_wait_time_ema
            : state->default_verify_wait_time;
    const double estimated_rtt = state->rtt_ema_ready
            ? state->rtt_ema
            : state->default_rtt;
    const double estimated_uplink = state->uplink_bytes_per_second_ema_ready
            ? state->uplink_bytes_per_second_ema
            : state->default_uplink_bytes_per_second;
    const double estimated_network_time = estimated_rtt;
    const double estimated_network_time_per_token = 4.0 / std::max(1.0, estimated_uplink);
    const double estimated_queue_time = (state->use_queue_in_goodput && state->queue_time_ema_ready)
            ? state->queue_time_ema
            : 0.0;
    bool used_prefetch = false;
    AdaptiveDraftResult draft_result;
    if (state->has_prefetch && !state->prefetch_drafts.empty()) {
        used_prefetch = true;
        draft_result.tokens = std::move(state->prefetch_drafts);
        draft_result.confidences = std::move(state->prefetch_confidences);
        draft_result.calibrated_confidences = std::move(state->prefetch_calibrated);
        state->has_prefetch = false;
        state->prefetch_drafts.clear();
        state->prefetch_confidences.clear();
        state->prefetch_calibrated.clear();
    } else {
        if (state->branch_kv) {
            llama_memory_t mem = llama_get_memory(context);
            if (!llama_memory_seq_rm(mem, 1, -1, -1)) {
                LOGe("hetero_phone branch_rm_fail");
                return env->NewStringUTF("");
            }
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
            state->branch_ready = true;
            LOGi("hetero_phone branch_open n_past=%d", state->n_past);
        }
        draft_result = generate_icn_multi_local_draft(
                *state,
                context,
                vocab,
                prompt_tgt,
                id_last,
                estimated_draft_per_token,
                estimated_verify_wait_time,
                estimated_network_time,
                estimated_network_time_per_token,
                estimated_queue_time);
    }
    llama_tokens draft_tokens = std::move(draft_result.tokens);
    std::vector<float> draft_confidences = std::move(draft_result.confidences);
    std::vector<float> calibrated_confidences = std::move(draft_result.calibrated_confidences);
    state->adaptive_n_draft = clamp_adaptive_n_draft(*state, static_cast<int>(draft_tokens.size()));
    if (draft_tokens.empty()) {
        LOGe("hetero_phone draft_empty last=%d n_past=%d",
                static_cast<int>(id_last),
                state->n_past);
        return env->NewStringUTF("");
    }
    LOGi("hetero_phone draft_ready count=%zu ids=%s",
            draft_tokens.size(),
            join_token_ids(draft_tokens).c_str());
    auto end1 = std::chrono::high_resolution_clock::now();
    double draft_time = std::chrono::duration<double>(end1 - start1).count();
    state->draft_times.push_back(draft_time);
    try {
        const uint32_t request_id = cloud_next_request_id(*state);
        const auto binary_msg = pack_token_request_v3(
                heterospec_v3::MSG_VERIFY_REQUEST,
                request_id,
                draft_tokens);
        const int payload_bytes = static_cast<int>(binary_msg.size());

        cloud_prepare_pending(*state, request_id);
        auto start = std::chrono::high_resolution_clock::now();

        state->client.send(state->connection,
            binary_msg.data(),
            binary_msg.size(),
            websocketpp::frame::opcode::binary);
        LOGi("hetero_phone verify_sent request_id=%u drafts=%s",
                request_id,
                join_token_ids(draft_tokens).c_str());

        const std::string icn_multi_icn_name = icn_multi_build_icn_name_v2(
                *state, prompt_tgt, draft_tokens, static_cast<int>(draft_tokens.size()));
        LOGi("icn_multi_icn_name=%s", icn_multi_icn_name.c_str());
        LOGi("icn_multi_raw_conf=%s icn_multi_calibrated_p=%s",
                icn_multi_join_floats(draft_confidences).c_str(),
                icn_multi_join_floats(calibrated_confidences).c_str());

        llama_tokens final_tokens;
        const int draft_count = static_cast<int>(draft_tokens.size());
        int accepted_draft_count = 0;
        double verify_time = 0.0;
        double cloud_verify_time = 0.0;
        double queue_time = 0.0;
        uint32_t logged_verify_us = 0;
        uint32_t logged_queue_us = 0;
        const llama_pos pos_base = static_cast<llama_pos>(prompt_tgt.size());
        OverlapPrefetch prefetch;
        if (!used_prefetch) {
            const int n_draft_max = clamp_adaptive_n_draft(
                    *state,
                    state->draft_strategy == DraftStrategy::Adaptive
                            ? state->max_n_draft
                            : CloudState::kFixedNDraft);
            overlap_while_waiting(
                    *state,
                    context,
                    vocab,
                    draft_tokens,
                    pos_base,
                    n_draft_max,
                    prefetch);
        }

        std::unique_lock<std::mutex> lock(state->mutex);
        // The next cloud forward folds the bonus token into the 27B cache before
        // it logs. A cold FP8 M-tile autotunes for about a minute; 10s drops the
        // socket and the UI stays on the token already shown.
        const bool ready = state->cv.wait_for(lock, std::chrono::seconds(120), [state] {
            return state->has_verify_response || state->has_error || !state->is_connected;
        });
        if (!ready) {
            LOGe("hetero_phone verify_timeout request_id=%u", request_id);
            state->pending_request_id = 0;
            state->is_connected = false;
            lock.unlock();
            state->client.stop_perpetual();
            state->client.stop();
            return nullptr;
        }
        if (state->has_error) {
            LOGe("hetero_phone verify_error request_id=%u code=%u message=%s",
                    state->error_request_id,
                    state->error_code,
                    state->error_message.c_str());
            return env->NewStringUTF("");
        }
        if (state->has_verify_response) {
            auto end = std::chrono::high_resolution_clock::now();
            verify_time = std::chrono::duration<double>(end - start).count();
            final_tokens = std::move(state->accepted_tokens);
            accepted_draft_count = std::max(0, std::min(state->response_accepted_draft_count, draft_count));
            logged_verify_us = state->response_verify_us;
            logged_queue_us = state->response_queue_us;
            cloud_verify_time = state->response_verify_us > 0
                    ? static_cast<double>(state->response_verify_us) / 1000000.0
                    : std::max(0.0, verify_time - estimated_network_time);
            queue_time = static_cast<double>(state->response_queue_us) / 1000000.0;
            state->accepted_tokens.clear();
            state->has_verify_response = false;
            state->response_accepted_draft_count = 0;
            state->response_verify_us = 0;
            state->response_queue_us = 0;
            LOGi("hetero_phone verify_ok request_id=%u accepted=%d wait_s=%.3f tokens=%s",
                    request_id,
                    accepted_draft_count,
                    verify_time,
                    join_token_ids(final_tokens).c_str());
        } else {
            LOGe("hetero_phone verify_missing request_id=%u connected=%d",
                    request_id,
                    state->is_connected ? 1 : 0);
            state->pending_request_id = 0;
        }
        lock.unlock();

        if (final_tokens.empty()) {
            LOGe("hetero_phone verify_empty request_id=%u", request_id);
            return env->NewStringUTF("");
        }

        const double observed_network_time = std::max(
                0.001,
                verify_time - cloud_verify_time - queue_time);
        const int n_out = static_cast<int>(final_tokens.size());
        const double round_s = draft_time + verify_time;
        double tpot_excl_ttft = 0.0;
        if (state->exclude_next_round_as_ttft) {
            state->exclude_next_round_as_ttft = false;
        } else if (n_out > 0) {
            tpot_excl_ttft = round_s / static_cast<double>(n_out);
        }
        if (logged_verify_us == 0 && cloud_verify_time > 0.0) {
            logged_verify_us = static_cast<uint32_t>(cloud_verify_time * 1000000.0 + 0.5);
        }
        LOGi("hetero_round strategy=%s k=%d accepted=%d draft_ms=%.3f verify_us=%u queue_us=%u rtt_ms=%.3f tpot_excl_ttft=%.6f",
                draft_strategy_name(state->draft_strategy),
                draft_count,
                accepted_draft_count,
                draft_time * 1000.0,
                logged_verify_us,
                logged_queue_us,
                observed_network_time * 1000.0,
                tpot_excl_ttft);
            update_adaptive_n_draft(
                    state,
                    draft_time,
                    draft_count,
                    cloud_verify_time,
                    accepted_draft_count,
                    draft_confidences,
                    calibrated_confidences,
                    observed_network_time,
                    queue_time,
                    payload_bytes);
            const llama_token commit_last = state->last_token;
            const int commit_pos0 = heterospec_hotpath::commit_pos0(
                    static_cast<int>(state->prompt_tokens.size()));
            state->prompt_tokens.insert(state->prompt_tokens.end(), state->last_token);
            state->last_token = final_tokens.back();
            state->prompt_tokens.insert(state->prompt_tokens.end(), final_tokens.begin(), final_tokens.end() - 1);
            state->n_past += final_tokens.size();
            if (state->branch_kv && !commit_to_seq0(
                    context,
                    *state,
                    commit_last,
                    draft_tokens,
                    accepted_draft_count,
                    commit_pos0)) {
                return env->NewStringUTF("");
            }

            const int32_t bonus_token = static_cast<int32_t>(final_tokens.back());
            const int32_t guessed_bonus = prefetch.has_guessed_bonus
                    ? static_cast<int32_t>(prefetch.guessed_bonus)
                    : -1;
            const bool hit = heterospec_hotpath::overlap_hit(
                    accepted_draft_count,
                    draft_count,
                    bonus_token,
                    guessed_bonus,
                    static_cast<int>(prefetch.drafts.size()));
            LOGi("hetero_phone kv hit=%d n_past=%d accepted=%d bonus=%d trim=%d",
                    hit ? 1 : 0,
                    state->n_past,
                    accepted_draft_count,
                    bonus_token,
                    (!hit && (prefetch.decoded_last_sent || prefetch.has_guessed_bonus)) ? 1 : 0);
            state->accepted_tokens.clear();
            if (hit && heterospec_hotpath::keep_draft_branch(hit)) {
                state->has_prefetch = true;
                state->prefetch_drafts = std::move(prefetch.drafts);
                state->prefetch_confidences = std::move(prefetch.confidences);
                state->prefetch_calibrated = std::move(prefetch.calibrated);
                state->branch_ready = state->branch_kv;
            } else {
                state->has_prefetch = false;
                state->prefetch_drafts.clear();
                state->prefetch_confidences.clear();
                state->prefetch_calibrated.clear();
                llama_memory_t mem = llama_get_memory(context);
                if (state->branch_kv) {
                    if (!llama_memory_seq_rm(mem, 1, -1, -1)) {
                        LOGe("hetero_phone branch_rm_fail");
                        return env->NewStringUTF("");
                    }
                    state->branch_ready = false;
                    LOGi("hetero_phone branch_drop");
                } else {
                    bool removed = true;
                    const bool prefetch_hole = heterospec_hotpath::prefetch_leaves_kv_hole(
                            used_prefetch,
                            accepted_draft_count,
                            draft_count);
                    if (prefetch_hole) {
                        removed = false;
                        LOGi("hetero_phone prefetch_hole accepted=%d", accepted_draft_count);
                    } else {
                        if (prefetch.decoded_last_sent || prefetch.has_guessed_bonus) {
                            if (!llama_memory_seq_rm(
                                    mem,
                                    0,
                                    static_cast<llama_pos>(heterospec_hotpath::overlap_kv_trim_pos(
                                            static_cast<int>(pos_base),
                                            accepted_draft_count)),
                                    -1)) {
                                removed = false;
                            }
                        }
                        if (!llama_memory_seq_rm(mem, 0, state->n_past + 1, -1)) {
                            removed = false;
                        }
                        if (!removed) {
                            LOGe("hetero_phone seq_rm_fail n_past=%d", state->n_past);
                        }
                    }
                    if (!removed) {
                        llama_memory_clear(mem, true);
                        if (!prefill_prefix(context, state->prompt_tokens)) {
                            LOGe("hetero_phone reprefix_fail n=%zu", state->prompt_tokens.size());
                            return env->NewStringUTF("");
                        }
                        LOGi("hetero_phone reprefix n=%zu", state->prompt_tokens.size());
                    }
                }
            }

            // 处理输出
            for (const auto& token : final_tokens) {
                const auto n_cur = env->CallIntMethod(intvar_ncur, la_int_var_value);

                if (llama_vocab_is_eog(vocab, token) || n_cur >= n_len) {
                    LOGi("hetero_phone stop eog=%d n_cur=%d n_len=%d token=%d",
                            llama_vocab_is_eog(vocab, token) ? 1 : 0,
                            n_cur,
                            n_len,
                            static_cast<int>(token));
                    if (!state->draft_times.empty()) {
                        double total_draft_time = 0.0;
                        for (double time : state->draft_times) {
                            total_draft_time += time;
                        }
                        double avg_draft_time = total_draft_time / state->draft_times.size();
                        LOGi("Average draft time: %.3fs over %zu iterations", avg_draft_time, state->draft_times.size());
                    }

                    if (is_valid_utf8(cached_token_chars.c_str())) {
                        cached_token_chars += "[eog]";
                        auto result = env->NewStringUTF(cached_token_chars.c_str());
                        LOGi("hetero_phone emit text=%s", cached_token_chars.c_str());
                        cached_token_chars.clear();
                        return result;
                    } else {
                        LOGe("hetero_phone stop_incomplete bytes=%zu", cached_token_chars.size());
                        return nullptr;
                    }
                }

                env->CallVoidMethod(intvar_ncur, la_int_var_inc);
                cached_token_chars += common_token_to_piece(context, token);
            }

            // 返回生成的文本
            if (is_valid_utf8(cached_token_chars.c_str())) {
                auto result = env->NewStringUTF(cached_token_chars.c_str());
                LOGi("hetero_phone emit text=%s", cached_token_chars.c_str());
                cached_token_chars.clear();
                return result;
            } else {
                LOGe("hetero_phone emit_incomplete bytes=%zu", cached_token_chars.size());
                return env->NewStringUTF("");
            }
    } catch (const std::exception& e) {
        LOGe("hetero_phone exception %s", e.what());
    }

    LOGe("hetero_phone return_empty");
    return env->NewStringUTF("");
}
