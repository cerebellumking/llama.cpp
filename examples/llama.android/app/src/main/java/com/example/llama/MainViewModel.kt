package com.example.llama

import android.llama.cpp.LLamaAndroid
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

// 消息类型枚举
enum class MessageType {
    USER,    // 用户输入
    SYSTEM,  // 系统输出
}

// 消息数据类
data class ChatMessage(
    val content: String,
    val type: MessageType
)

class MainViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf(ChatMessage("Initializing...", MessageType.SYSTEM)))
        private set

    var message by mutableStateOf("")
        private set

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

        // 添加用户消息
        messages += ChatMessage(text, MessageType.USER)
        // 添加空的系统消息，用于接收输出
        messages += ChatMessage("", MessageType.SYSTEM)

        viewModelScope.launch {
            llamaAndroid.send(text)
                .catch {
                    Log.e(tag, "send() failed", it)
                    messages += ChatMessage(it.message!!, MessageType.SYSTEM)
                }
                .collect { 
                    // 更新最后一条系统消息
                    val lastMessage = messages.last()
                    messages = messages.dropLast(1) + ChatMessage(lastMessage.content + it, MessageType.SYSTEM)
                }
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

    fun load(pathToModel: String) {
        viewModelScope.launch {
            try {
                // 先卸载当前模型
                try {
                    llamaAndroid.unload()
                } catch (e: IllegalStateException) {
                    // 忽略卸载错误
                }
                // 加载新模型
                llamaAndroid.load(pathToModel)
                messages += ChatMessage("已切换到模型：$pathToModel", MessageType.SYSTEM)
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
    }

    fun log(message: String) {
        messages += ChatMessage(message, MessageType.SYSTEM)
    }
}
