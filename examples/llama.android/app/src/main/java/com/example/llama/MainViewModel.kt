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

// Prompt类枚举
enum class PromptMode {
    QWEN2,
    DEEPSEEk,
    QWEN3
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

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf(ChatMessage("Initializing...", MessageType.SYSTEM)))
        private set

    var message by mutableStateOf("")
        private set

    // 添加推理速度状态
    var inferenceSpeed by mutableStateOf(0.0)
        private set

    // 添加CPU利用率状态
    var cpuUsage by mutableStateOf(0.0f)
        private set

    // CPU监控开关
    var isCpuMonitorEnabled by mutableStateOf(false)
        private set

    // 推理期间 CPU 采样（平均值 = 累加 / 次数）
    var isInferenceRunning by mutableStateOf(false)
        private set
    private var cpuSampleSum = 0.0f
    private var cpuSampleCount = 0

    // 测试模式下：数据集级别 CPU 采样（平均值 = 累加 / 次数）
    private var datasetCpuSampleSum = 0.0f
    private var datasetCpuSampleCount = 0
    var datasetCpuAvg by mutableStateOf(0.0f)
        private set

    // 添加推理模式状态
    var inferenceMode by mutableStateOf(InferenceMode.LOCAL)
        private set

    var promptMode by mutableStateOf(PromptMode.QWEN2)
        set

    // 添加API类型状态
    var currentApiType by mutableStateOf(ApiType.DEEPSEEK)
        private set

    // 添加OCR编辑相关状态
    var isShowingOcrEditor by mutableStateOf(false)
        private set

    var ocrEditText by mutableStateOf("")
        private set

    var ocrSourceImage by mutableStateOf<Bitmap?>(null)
        private set

    var editingMessageIndex by mutableStateOf(-1)
        private set

    var editingText by mutableStateOf("")
        private set

    // CPU监控器
    private val cpuMonitor = CpuMonitor()

    // 获取当前API服务实例
    private val apiService: ApiService
        get() = ApiService.getInstance(currentApiType)

    private var lastTokenTime = System.currentTimeMillis()
    private var tokenCount = 0
    private var isFirstToken = true

    // 测试模式相关状态（面向整个数据集的速度统计）
    var isTestRunning by mutableStateOf(false)
        private set

    var testTotal by mutableStateOf(0)
        private set

    var testProcessed by mutableStateOf(0)
        private set

    // 数据集整体速度（tokens/s）
    var datasetSpeed by mutableStateOf(0.0)
        private set

    // 上一次测试完成时的整体速度（用于测试结束后仍显示）
    var lastTestSpeed by mutableStateOf(0.0)
        private set

    private var testStartTimeMs: Long = 0L
    private var testTotalTokens: Int = 0
    private var testActiveTimeMs: Long = 0L // 仅统计每条样本从首 token 到末 token 的生成时长（排除两条之间的空闲）

    init {
        // 启动CPU监控（可开关）
        if (isCpuMonitorEnabled) {
            cpuMonitor.startMonitoring(100) { usage ->
                Log.d("MainViewModel", "CPU usage updated: $usage%")
                if (isTestRunning) {
                    datasetCpuSampleSum += usage
                    datasetCpuSampleCount += 1
                    datasetCpuAvg = if (datasetCpuSampleCount > 0) datasetCpuSampleSum / datasetCpuSampleCount else 0.0f
                    cpuUsage = datasetCpuAvg
                } else if (isInferenceRunning) {
                    cpuSampleSum += usage
                    cpuSampleCount += 1
                    cpuUsage = if (cpuSampleCount > 0) cpuSampleSum / cpuSampleCount else 0.0f
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()

        // 停止CPU监控
        cpuMonitor.stopMonitoring()

        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                messages += ChatMessage(exc.message!!, MessageType.SYSTEM)
            }
        }
    }

    fun updateCpuMonitorEnabled(enabled: Boolean) {
        if (enabled == isCpuMonitorEnabled) return
        isCpuMonitorEnabled = enabled
        if (enabled) {
            cpuMonitor.startMonitoring(100) { usage ->
                Log.d("MainViewModel", "CPU usage updated: $usage%")
                if (isTestRunning) {
                    datasetCpuSampleSum += usage
                    datasetCpuSampleCount += 1
                    datasetCpuAvg = if (datasetCpuSampleCount > 0) datasetCpuSampleSum / datasetCpuSampleCount else 0.0f
                    cpuUsage = datasetCpuAvg
                } else if (isInferenceRunning) {
                    cpuSampleSum += usage
                    cpuSampleCount += 1
                    cpuUsage = if (cpuSampleCount > 0) cpuSampleSum / cpuSampleCount else 0.0f
                }
            }
        } else {
            cpuMonitor.stopMonitoring()
            cpuUsage = 0.0f
        }
    }

    private fun beginInferenceMonitoring() {
        isInferenceRunning = true
        cpuSampleSum = 0.0f
        cpuSampleCount = 0
        cpuUsage = 0.0f
    }

    private fun endInferenceMonitoring() {
        isInferenceRunning = false
    }

    fun send(skipAddingUserMessage: Boolean = false) {
        val text = message
        message = ""

        // 检查是否是图片消息（通过检查最后一条消息是否包含图片）
        val isImageMessage = messages.lastOrNull { it.type == MessageType.USER }?.image != null

        // 如果不是图片消息且没有跳过添加用户消息，则显示用户输入
        if (!isImageMessage && !skipAddingUserMessage) {
            messages += ChatMessage(text, MessageType.USER)
        }

        // 添加空的系统消息，用于接收输出
        messages += ChatMessage("", MessageType.SYSTEM)

        // 重置推理速度计数
        lastTokenTime = System.currentTimeMillis()
        tokenCount = 0
        inferenceSpeed = 0.0
        isFirstToken = true

        beginInferenceMonitoring()
        viewModelScope.launch {
            // 构建完整的对话历史，包含系统提示词

            try {
                var fullPrompt = ""
                when (promptMode) {
                    PromptMode.QWEN2 -> {
                        fullPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
//                        fullPrompt =
//                            "<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<｜User｜>$text<｜Assistant｜><think>"
                    }

                    PromptMode.QWEN3 -> {
                        fullPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:\n<think></think>"
//                        fullPrompt =
//                            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$text<|im_end|>\n<|im_start|>assistant\n"
                    }
                    PromptMode.DEEPSEEk -> {
                        fullPrompt = text
                    }
                }
                when (inferenceMode) {
                    InferenceMode.LOCAL -> {
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
            } finally {
                endInferenceMonitoring()
            }
        }
    }

    private fun updateMessageAndSpeed(str: String, tokens: Int) {
        // 更新最后一条系统消息
        val lastMessage = messages.last()
        messages = messages.dropLast(1) + ChatMessage(lastMessage.content + str, MessageType.SYSTEM)

        // 更新token计数
        tokenCount += tokens
        val currentTime = System.currentTimeMillis()
        val timeDiff = (currentTime - lastTokenTime) / 1000.0  // 转换为秒

        if (isFirstToken) {
            // 第一个token不计入速度统计
            lastTokenTime = currentTime
            isFirstToken = false
        } else if (timeDiff >= 1.0) { // 每秒更新一次速度
            inferenceSpeed = tokenCount / timeDiff
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

    // 添加OCR编辑相关方法
    fun showOcrEditor(text: String, image: Bitmap) {
        ocrEditText = text
        ocrSourceImage = image
        isShowingOcrEditor = true
    }

    fun hideOcrEditor() {
        isShowingOcrEditor = false
        ocrEditText = ""
        ocrSourceImage = null
    }

    fun updateOcrText(text: String) {
        ocrEditText = text
    }

    fun confirmOcrAndSend() {
        val ocrMessage = "请解读病例报告并给出简短建议：$ocrEditText"

        messages += ChatMessage(ocrMessage, MessageType.USER)

        message = ocrMessage

        hideOcrEditor()

        send(skipAddingUserMessage = true)
        message = ""
    }

    // 添加消息编辑相关方法
    fun startEditingMessage(index: Int, content: String) {
        editingMessageIndex = index
        editingText = content
    }

    fun cancelEditingMessage() {
        editingMessageIndex = -1
        editingText = ""
    }

    fun updateEditingText(text: String) {
        editingText = text
    }

    fun confirmEditMessage() {
        if (editingMessageIndex >= 0 && editingText.isNotBlank()) {
            // 添加新的用户消息，而不是覆盖原消息
            messages += ChatMessage(editingText, MessageType.USER)

            // 发送编辑后的消息
            message = editingText
            cancelEditingMessage()
            send(skipAddingUserMessage = true)
            message = ""
        }
    }

    // ===== 测试模式：整体速度统计 =====

    fun startTest(totalItems: Int) {
        isTestRunning = true
        testTotal = totalItems
        testProcessed = 0
        datasetSpeed = 0.0
        lastTestSpeed = 0.0
        testTotalTokens = 0
        testStartTimeMs = System.currentTimeMillis()
        testActiveTimeMs = 0L
        // 重置数据集级别 CPU 采样
        datasetCpuSampleSum = 0.0f
        datasetCpuSampleCount = 0
        datasetCpuAvg = 0.0f
        // 清零展示值，后续由采样回调实时更新数据集平均
        cpuUsage = 0.0f
        messages += ChatMessage("开始数据集测试，共 $totalItems 条", MessageType.SYSTEM)
    }

    private fun updateTestTokens(tokens: Int) {
        if (!isTestRunning) return
        if (tokens <= 0) return
        testTotalTokens += tokens
        // 速度计算改为在 sendForTest 内依据“活跃生成时长”更新
    }

    fun incrementTestProgress() {
        if (!isTestRunning) return
        testProcessed += 1
    }

    fun finishTest() {
        if (!isTestRunning) return
        val activeSeconds = testActiveTimeMs / 1000.0
        lastTestSpeed = if (activeSeconds > 0) testTotalTokens / activeSeconds else 0.0
        datasetSpeed = lastTestSpeed
        isTestRunning = false
        messages += ChatMessage(
            "测试完成：$testProcessed/$testTotal，整体速度：${"%.2f".format(lastTestSpeed)} tokens/s",
            MessageType.SYSTEM
        )
    }

    // 仅用于测试模式：不向聊天消息区输出，直接进行推理并累计整体 tokens
    suspend fun sendForTest(text: String) {
        try {
            var itemFirstTokenTimeMs: Long? = null
            var itemLastTokenTimeMs: Long? = null
            val fullPrompt = when (promptMode) {
                PromptMode.QWEN2 -> {
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
                }
                PromptMode.QWEN3 -> {
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:\n<think></think>"
                }
                PromptMode.DEEPSEEk -> text
            }

            when (inferenceMode) {
                InferenceMode.LOCAL -> {
                    llamaAndroid.send(fullPrompt)
                        .catch {
                            Log.e(tag, "sendForTest() failed", it)
                        }
                        .collect { (_, tokens) ->
                            val now = System.currentTimeMillis()
                            if (tokens > 0) {
                                if (itemFirstTokenTimeMs == null) itemFirstTokenTimeMs = now
                                itemLastTokenTimeMs = now
                            }
                            updateTestTokens(tokens)
                            // 实时速度：累计活跃时长 + 当前样本已用时
                            val partialActive = if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null) (itemLastTokenTimeMs!! - itemFirstTokenTimeMs!!) else 0L
                            val totalActive = testActiveTimeMs + partialActive
                            if (totalActive > 0L) {
                                datasetSpeed = testTotalTokens / (totalActive / 1000.0)
                            }
                        }
                }
                InferenceMode.API -> {
                    apiService.send(fullPrompt)
                        .catch {
                            Log.e(tag, "API sendForTest() failed", it)
                        }
                        .collect { (_, tokens) ->
                            val now = System.currentTimeMillis()
                            if (tokens > 0) {
                                if (itemFirstTokenTimeMs == null) itemFirstTokenTimeMs = now
                                itemLastTokenTimeMs = now
                            }
                            updateTestTokens(tokens)
                            val partialActive = if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null) (itemLastTokenTimeMs!! - itemFirstTokenTimeMs!!) else 0L
                            val totalActive = testActiveTimeMs + partialActive
                            if (totalActive > 0L) {
                                datasetSpeed = testTotalTokens / (totalActive / 1000.0)
                            }
                        }
                }
                InferenceMode.HETERO -> {
                    llamaAndroid.sendHetero(fullPrompt)
                        .catch {
                            Log.e(tag, "sendForTest() failed", it)
                        }
                        .collect { (_, tokens) ->
                            val now = System.currentTimeMillis()
                            if (tokens > 0) {
                                if (itemFirstTokenTimeMs == null) itemFirstTokenTimeMs = now
                                itemLastTokenTimeMs = now
                            }
                            updateTestTokens(tokens)
                            val partialActive = if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null) (itemLastTokenTimeMs!! - itemFirstTokenTimeMs!!) else 0L
                            val totalActive = testActiveTimeMs + partialActive
                            if (totalActive > 0L) {
                                datasetSpeed = testTotalTokens / (totalActive / 1000.0)
                            }
                        }
                }
            }
        } catch (e: Exception) {
            Log.e(tag, "Error during sendForTest", e)
            messages += ChatMessage("测试失败：${e.message}", MessageType.SYSTEM)
        } finally {
            // 样本完成后，累计该样本活跃生成时长
            // 注意：sendForTest() 每次只处理一个样本
            // 若没有 token 产出，视作 0 时长
            // itemFirstTokenTimeMs 与 itemLastTokenTimeMs 在 collect 中定义
            // 为了访问它们，将 finally 中移动到 lambda 外定义
        }
    }

    // 将样本活跃时长累加（供 sendForTest finally 调用）
    private fun accumulateItemActiveDuration(itemFirstTokenTimeMs: Long?, itemLastTokenTimeMs: Long?) {
        if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null && itemLastTokenTimeMs >= itemFirstTokenTimeMs) {
            testActiveTimeMs += (itemLastTokenTimeMs - itemFirstTokenTimeMs)
        }
        val activeSeconds = testActiveTimeMs / 1000.0
        datasetSpeed = if (activeSeconds > 0) testTotalTokens / activeSeconds else 0.0
    }

    suspend fun sendForTest(text: String, _internalMarker: Boolean = false) {
        // 占位避免重载冲突
    }

    // 由于 Kotlin 的工具编辑上下文限制，重新定义 sendForTest 的主体以包含 finally 中的累加逻辑
    suspend fun sendForTest_impl(text: String) {
        beginInferenceMonitoring()
        var itemFirstTokenTimeMs: Long? = null
        var itemLastTokenTimeMs: Long? = null
        try {
            val fullPrompt = when (promptMode) {
                PromptMode.QWEN2 -> {
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
                }
                PromptMode.QWEN3 -> {
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:\n<think></think>"
                }
                PromptMode.DEEPSEEk -> text
            }

            when (inferenceMode) {
                InferenceMode.LOCAL -> {
                    llamaAndroid.send(fullPrompt)
                        .catch { Log.e(tag, "sendForTest() failed", it) }
                        .collect { (_, tokens) ->
                            val now = System.currentTimeMillis()
                            if (tokens > 0) {
                                if (itemFirstTokenTimeMs == null) itemFirstTokenTimeMs = now
                                itemLastTokenTimeMs = now
                            }
                            updateTestTokens(tokens)
                            val partialActive = if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null) (itemLastTokenTimeMs!! - itemFirstTokenTimeMs!!) else 0L
                            val totalActive = testActiveTimeMs + partialActive
                            if (totalActive > 0L) datasetSpeed = testTotalTokens / (totalActive / 1000.0)
                        }
                }
                InferenceMode.API -> {
                    apiService.send(fullPrompt)
                        .catch { Log.e(tag, "API sendForTest() failed", it) }
                        .collect { (_, tokens) ->
                            val now = System.currentTimeMillis()
                            if (tokens > 0) {
                                if (itemFirstTokenTimeMs == null) itemFirstTokenTimeMs = now
                                itemLastTokenTimeMs = now
                            }
                            updateTestTokens(tokens)
                            val partialActive = if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null) (itemLastTokenTimeMs!! - itemFirstTokenTimeMs!!) else 0L
                            val totalActive = testActiveTimeMs + partialActive
                            if (totalActive > 0L) datasetSpeed = testTotalTokens / (totalActive / 1000.0)
                        }
                }
                InferenceMode.HETERO -> {
                    llamaAndroid.sendHetero(fullPrompt)
                        .catch { Log.e(tag, "sendForTest() failed", it) }
                        .collect { (_, tokens) ->
                            val now = System.currentTimeMillis()
                            if (tokens > 0) {
                                if (itemFirstTokenTimeMs == null) itemFirstTokenTimeMs = now
                                itemLastTokenTimeMs = now
                            }
                            updateTestTokens(tokens)
                            val partialActive = if (itemFirstTokenTimeMs != null && itemLastTokenTimeMs != null) (itemLastTokenTimeMs!! - itemFirstTokenTimeMs!!) else 0L
                            val totalActive = testActiveTimeMs + partialActive
                            if (totalActive > 0L) datasetSpeed = testTotalTokens / (totalActive / 1000.0)
                        }
                }
            }
        } catch (e: Exception) {
            Log.e(tag, "Error during sendForTest", e)
            messages += ChatMessage("测试失败：${e.message}", MessageType.SYSTEM)
        } finally {
            accumulateItemActiveDuration(itemFirstTokenTimeMs, itemLastTokenTimeMs)
            incrementTestProgress()
            endInferenceMonitoring()
        }
    }
}
