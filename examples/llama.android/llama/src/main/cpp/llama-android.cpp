#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include "llama.h"
#include "common.h"
#include "speculative.h"
#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_client.hpp"
#include "json.hpp"
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

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("Loading model from %s", path_to_model);

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

    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    LOGi("Using %d threads", n_threads);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = 4096;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_batch        = 2048;  // 增加批处理大小

    llama_context * context = llama_init_from_model(model, ctx_params);

    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
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
Java_android_llama_cpp_LLamaAndroid_new_1sampler(JNIEnv *, jobject) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1sampler(JNIEnv *, jobject, jlong sampler_pointer) {
    llama_sampler_free(reinterpret_cast<llama_sampler *>(sampler_pointer));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1init(JNIEnv *, jobject) {
    llama_backend_init();
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

    LOGi("n_len = %d, n_ctx = %d, n_kv_req = %d", n_len, n_ctx, n_kv_req);

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

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetMethodID(la_int_var, "getValue", "()I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

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
    llama_kv_self_clear(reinterpret_cast<llama_context *>(context));
}

// HeteroSpec

typedef websocketpp::client<websocketpp::config::asio_client> ws_client;

struct CloudState {
    ws_client client;
    websocketpp::connection_hdl connection;
    std::string server_url;
    bool is_connected = false;
    std::vector<int> accepted_tokens;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<llama_token> prompt_tokens;
    int n_past{};
    llama_token last_token{};
    std::thread client_thread;
    std::condition_variable connect_cv;
    bool connection_ready = false;
};

// 修改 WebSocket 初始化函数
bool init_websocket(CloudState& state, const std::string& url) {
    if (state.is_connected) return true;

    // 优化日志开销
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

    state.client.set_message_handler([&state](websocketpp::connection_hdl hdl, ws_client::message_ptr msg) {
        try {
            const auto* data = reinterpret_cast<const uint8_t*>(msg->get_payload().data());
            const size_t size = msg->get_payload().size();

            if (size > 0 && data[0] == 0x01) {  // verify消息
                std::vector<int> tokens;
                tokens.reserve((size - 1) / sizeof(int));

                const int* token_ptr = reinterpret_cast<const int*>(data + 1);
                const size_t num_tokens = (size - 1) / sizeof(int);
                tokens.assign(token_ptr, token_ptr + num_tokens);

                std::unique_lock<std::mutex> lock(state.mutex);
                state.accepted_tokens = std::move(tokens);
                state.cv.notify_one();  // 使用notify_one替代notify_all
            }
        } catch (const std::exception& e) {
            LOGe("Error parsing message: %s", e.what());
        }
    });

    state.client.set_open_handler([&state](websocketpp::connection_hdl hdl) {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.connection_ready = true;
        state.connect_cv.notify_all();
        LOGi("WebSocket connection opened");
    });

    state.client.init_asio();

    websocketpp::lib::error_code ec;
    auto con = state.client.get_connection(url, ec);
    if (ec) {
        LOGe("Failed to create connection: %s", ec.message().c_str());
        return false;
    }

    state.client.connect(con);

    // 在单独的线程中运行客户端
    state.client_thread = std::thread([&state]() {
        try {
            state.client.run();
        } catch (const std::exception& e) {
            LOGe("WebSocket client thread error: %s", e.what());
        }
    });

    {
        std::unique_lock<std::mutex> lock(state.mutex);
        if (!state.connect_cv.wait_for(lock, std::chrono::seconds(10), [&state] {
            return state.connection_ready;
        })) {
            LOGe("WebSocket connection timeout");
            return false;
        }
    }

    state.is_connected = true;
    state.connection = con->get_handle();
    return true;
}

// 添加清理函数
extern "C" JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_heterospec_1cleanup(JNIEnv *env, jobject thiz) {
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeStatePtr", "J");
    auto* state = reinterpret_cast<CloudState*>(env->GetLongField(thiz, fid));

    if (state) {
        if (state->is_connected) {
            state->client.stop();
            if (state->client_thread.joinable()) {
                state->client_thread.join();
            }
        }
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
        jstring server_url)
{
    cached_token_chars.clear();
    const auto text = env->GetStringUTFChars(jtext, 0);
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    bool parse_special = (format_chat == JNI_TRUE);
    const auto tokens_list = common_tokenize(context, text, true, parse_special);

    const char *url = env->GetStringUTFChars(server_url, nullptr);
    auto* state = new CloudState();
    state->server_url = url;

    // 将状态指针保存到Java对象中
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeStatePtr", "J");
    env->SetLongField(thiz, fid, reinterpret_cast<jlong>(state));

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

    // 发送prefill请求
    try {
        if (!state->is_connected || !state->connection_ready) {
            LOGe("WebSocket not connected or not ready");
            env->ReleaseStringUTFChars(server_url, url);
            env->ReleaseStringUTFChars(jtext, text);
            return -2;
        }

        // 准备二进制消息
        std::vector<uint8_t> binary_msg;
        binary_msg.reserve(1 + tokens_list.size() * sizeof(llama_token));  // 预分配空间
        binary_msg.push_back(0x00);  // 消息类型：prefill

        // 直接复制token数据
        const size_t data_size = tokens_list.size() * sizeof(llama_token);
        binary_msg.resize(1 + data_size);
        memcpy(binary_msg.data() + 1, tokens_list.data(), data_size);

        state->client.send(state->connection, binary_msg.data(), binary_msg.size(), websocketpp::frame::opcode::binary);
        LOGi("Prefill message sent successfully");
    } catch (const std::exception& e) {
        LOGe("Exception while sending message: %s", e.what());
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

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetMethodID(la_int_var, "getValue", "()I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

    if (!state || !state->is_connected) {
        return env->NewStringUTF("");
    }

    // 1. 本地生成推测token (使用llama.cpp的API)
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);

    struct common_speculative_params params_spec;
    params_spec.n_draft = 3;
    params_spec.n_reuse = llama_n_ctx(context) - params_spec.n_draft;
    params_spec.p_min   = 0;
    struct common_speculative * spec = common_speculative_init(context);

    // 使用保存的状态
    const auto& prompt_tgt = state->prompt_tokens;
    const auto id_last = state->last_token;

    // 生成草稿token
    auto start1 = std::chrono::high_resolution_clock::now();
    llama_tokens draft_tokens = speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);
    auto end1 = std::chrono::high_resolution_clock::now();
    double draft_time = std::chrono::duration<double>(end1 - start1).count() / params_spec.n_draft;
    LOGi("Draft generation time: %.3fs, generated %d tokens", draft_time, draft_tokens.size());

    try {
        // 准备二进制消息
        std::vector<uint8_t> binary_msg;
        binary_msg.reserve(1 + draft_tokens.size() * sizeof(llama_token));  // 预分配空间
        binary_msg.push_back(0x01);  // 消息类型：verify

        // 直接复制token数据
        const size_t data_size = draft_tokens.size() * sizeof(llama_token);
        binary_msg.resize(1 + data_size);
        memcpy(binary_msg.data() + 1, draft_tokens.data(), data_size);

        auto start = std::chrono::high_resolution_clock::now();

        // 发送消息并等待响应
        state->client.send(state->connection,
            binary_msg.data(),
            binary_msg.size(),
            websocketpp::frame::opcode::binary);

        std::unique_lock<std::mutex> lock(state->mutex);
        if (state->cv.wait_for(lock, std::chrono::milliseconds(500), [state] {
            return !state->accepted_tokens.empty();
        })) {
            auto end = std::chrono::high_resolution_clock::now();
            double verify_time = std::chrono::duration<double>(end - start).count();
            LOGi("Verification time: %.3fs", verify_time);

            llama_tokens final_tokens = std::move(state->accepted_tokens);
            state->prompt_tokens.insert(state->prompt_tokens.end(), state->last_token);
            state->last_token = final_tokens.back();
            state->prompt_tokens.insert(state->prompt_tokens.end(), final_tokens.begin(), final_tokens.end() - 1);
            state->n_past += final_tokens.size();

            // 重置状态
            state->accepted_tokens.clear();
            llama_kv_self_seq_rm(context, 0, state->n_past + 1, -1);

            // 处理输出
            for (const auto& token : final_tokens) {
                const auto n_cur = env->CallIntMethod(intvar_ncur, la_int_var_value);

                if (llama_vocab_is_eog(vocab, token) || n_cur >= n_len) {
                    return nullptr;
                }

                env->CallVoidMethod(intvar_ncur, la_int_var_inc);
                cached_token_chars += common_token_to_piece(context, token);
            }

            // 返回生成的文本
            if (is_valid_utf8(cached_token_chars.c_str())) {
                auto result = env->NewStringUTF(cached_token_chars.c_str());
                LOGi("Generated text: %s", cached_token_chars.c_str());
                cached_token_chars.clear();
                return result;
            } else {
                return env->NewStringUTF("");
            }
        }
    } catch (const std::exception& e) {
        LOGe("Exception in heterospec_loop: %s", e.what());
    }

    return env->NewStringUTF("");
}
