package com.example.llama

import android.graphics.Bitmap
import android.llama.cpp.LLamaAndroid
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.api.ApiService
import com.example.llama.api.ApiType
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

// 消息类型枚举
enum class MessageType {
    USER,    // 用户输入
    SYSTEM,  // 系统输出
}

// 推理模式枚举
enum class InferenceMode {
    LOCAL,   // 本地推理
    API,     // API推理
    HETERO   // 异构推理
}

// 消息数据类
data class ChatMessage(
    val content: String,
    val type: MessageType,
    val image: Bitmap? = null
)

class MainViewModel(
    private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()
): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf(ChatMessage("Initializing...", MessageType.SYSTEM)))
        private set

    var message by mutableStateOf("")
        private set

    // 添加推理速度状态
    var inferenceSpeed by mutableStateOf(0.0)
        private set

    // 添加推理模式状态
    var inferenceMode by mutableStateOf(InferenceMode.LOCAL)
        private set

    // 添加API类型状态
    var currentApiType by mutableStateOf(ApiType.DEEPSEEK)
        private set

    // 获取当前API服务实例
    private val apiService: ApiService
        get() = ApiService.getInstance(currentApiType)

    private var lastTokenTime = System.nanoTime()
    private var tokenCount = 0
    private var isFirstToken = true

    override fun onCleared() {
        super.onCleared()

        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                messages += ChatMessage(exc.message!!, MessageType.SYSTEM)
            }
        }
    }

    fun send() {
        val text = message
        message = ""

        // 检查是否是图片消息（通过检查最后一条消息是否包含图片）
        val isImageMessage = messages.lastOrNull { it.type == MessageType.USER }?.image != null

        // 如果不是图片消息，则显示用户输入
        if (!isImageMessage) {
            messages += ChatMessage(text, MessageType.USER)
        }

        // 添加空的系统消息，用于接收输出
        messages += ChatMessage("", MessageType.SYSTEM)

        // 重置推理速度计数
        lastTokenTime = System.nanoTime()
        tokenCount = 0
        inferenceSpeed = 0.0
        isFirstToken = true

        viewModelScope.launch {
            // 构建完整的对话历史，包含系统提示词

            try {
                when (inferenceMode) {
                    InferenceMode.LOCAL -> {
                        val fullPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
                        llamaAndroid.send(fullPrompt)
                            .catch {
                                Log.e(tag, "send() failed", it)
                                messages += ChatMessage(it.message!!, MessageType.SYSTEM)
                            }
                            .collect { (str, tokens) ->
                                // 更新最后一条系统消息
                                updateMessageAndSpeed(str, tokens)
                            }
                    }
                    InferenceMode.API -> {
                        val fullPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
                        apiService.send(fullPrompt)
                            .catch {
                                Log.e(tag, "API send() failed", it)
                                messages += ChatMessage(it.message!!, MessageType.SYSTEM)
                            }
                            .collect { (str, tokens) ->
                                // 更新最后一条系统消息
                                updateMessageAndSpeed(str, tokens)
                            }
                    }
                    InferenceMode.HETERO -> {
                        val fullPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
                        llamaAndroid.sendHetero(fullPrompt)
                            .catch {
                                Log.e(tag, "send() failed", it)
                                messages += ChatMessage(it.message!!, MessageType.SYSTEM)
                            }
                            .collect { (str, tokens) ->
                                // 更新最后一条系统消息
                                updateMessageAndSpeed(str, tokens)
                            }
                    }
                }
            } catch (e: Exception) {
                Log.e(tag, "Error during inference", e)
                messages += ChatMessage("Error: ${e.message}", MessageType.SYSTEM)
            }
        }
    }

    private fun updateMessageAndSpeed(str: String, tokens: Int) {
        // 更新最后一条系统消息
        val lastMessage = messages.last()
        messages = messages.dropLast(1) + ChatMessage(lastMessage.content + str, MessageType.SYSTEM)

        // 更新token计数
        tokenCount += tokens
        val currentTime = System.nanoTime()
        val timeDiff = (currentTime - lastTokenTime) / NanosPerSecond

        if (isFirstToken) {
            // 第一个token不计入速度统计
            lastTokenTime = currentTime
            isFirstToken = false
        } else if (timeDiff >= 1.0) { // 每秒更新一次速度
            inferenceSpeed = tokenCount / timeDiff
            lastTokenTime = currentTime
            tokenCount = 0
        }
    }

    fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
        viewModelScope.launch {
            try {
                val start = System.nanoTime()
                val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
                val end = System.nanoTime()

                messages += ChatMessage(warmupResult, MessageType.SYSTEM)

                val warmup = (end - start).toDouble() / NanosPerSecond
                messages += ChatMessage("Warm up time: $warmup seconds, please wait...", MessageType.SYSTEM)

                if (warmup > 5.0) {
                    messages += ChatMessage("Warm up took too long, aborting benchmark", MessageType.SYSTEM)
                    return@launch
                }

                messages += ChatMessage(llamaAndroid.bench(512, 128, 1, 3), MessageType.SYSTEM)
            } catch (exc: IllegalStateException) {
                Log.e(tag, "bench() failed", exc)
                messages += ChatMessage(exc.message!!, MessageType.SYSTEM)
            }
        }
    }

    fun load(pathToModel: String, isHetero: Boolean = false) {
        viewModelScope.launch {
            try {
                // 切换到本地模式
                if(isHetero) {
                    inferenceMode = InferenceMode.HETERO
                } else {
                    inferenceMode = InferenceMode.LOCAL
                }

                // 先卸载当前模型
                try {
                    llamaAndroid.unload()
                } catch (e: IllegalStateException) {
                    // 忽略卸载错误
                }
                // 加载新模型
                llamaAndroid.load(pathToModel)
                val fileName = pathToModel.substringAfterLast("/")
                if(isHetero){
                    messages += ChatMessage("使用草稿模型：$fileName", MessageType.SYSTEM)
                } else {
                    messages += ChatMessage("已切换到模型：$fileName", MessageType.SYSTEM)
                }
            } catch (exc: IllegalStateException) {
                Log.e(tag, "load() failed", exc)
                messages += ChatMessage(exc.message!!, MessageType.SYSTEM)
            }
        }
    }

    fun updateMessage(newMessage: String) {
        message = newMessage
    }

    fun clear() {
        messages = listOf(ChatMessage("已清除对话历史", MessageType.SYSTEM))
        inferenceSpeed = 0.0
    }

    fun log(message: String) {
        messages += ChatMessage(message, MessageType.SYSTEM)
    }

    fun switchToApiMode(type: ApiType) {
        currentApiType = type
        inferenceMode = InferenceMode.API
        messages += ChatMessage("已切换到 ${type.name} API 模式", MessageType.SYSTEM)
    }

    fun addImageMessage(bitmap: Bitmap) {
        messages += ChatMessage("", MessageType.USER, bitmap)
    }
}
