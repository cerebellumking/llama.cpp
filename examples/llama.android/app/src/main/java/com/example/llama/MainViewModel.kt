package com.example.llama

import android.graphics.Bitmap
import android.llama.cpp.LLamaAndroid
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.api.ApiService
import com.example.llama.api.ApiType
import com.example.llama.api.DeepseekApiService
import com.example.llama.api.QwenApiService
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

private const val TEST_API_MAX_NEW_TOKENS = 256

class MainViewModel(
    private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()
) : ViewModel() {

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(emptyList<ChatMessage>())
        private set

    var message by mutableStateOf("")
        private set

    var inferenceSpeed by mutableStateOf(0.0)
        private set

    var requestTpotMs by mutableStateOf(-1.0)
        private set

    var cpuUsage by mutableStateOf(-1.0f)
        private set

    var isCpuMonitorEnabled by mutableStateOf(false)
        private set

    var hasProcessCpuSample by mutableStateOf(false)
        private set

    var isInferenceRunning by mutableStateOf(false)
        private set

    private var datasetCpuMs = 0L
    private var datasetWallMs = 0L
    var datasetCpuAvg by mutableStateOf(-1.0f)
        private set
    private val cpuCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
    private var testOwnsMonitor = false

    var inferenceMode by mutableStateOf(InferenceMode.LOCAL)
        private set

    var promptMode by mutableStateOf(PromptMode.QWEN2)

    var currentApiType by mutableStateOf(ApiType.DEEPSEEK)
        private set

    private var selectedApiName: String = "DeepSeek API"

    var heteroServerUrl by mutableStateOf(LLamaAndroid.DEFAULT_HETERO_SERVER_URL)
        private set

    var heteroDraftStrategy by mutableStateOf(LLamaAndroid.DRAFT_STRATEGY_FIXED3)
        private set

    var isShowingOcrEditor by mutableStateOf(false)
        private set

    var ocrEditText by mutableStateOf("")
        private set

    var ocrSourceImage by mutableStateOf<Bitmap?>(null)
        private set

    var editingMessageId by mutableStateOf<String?>(null)
        private set

    var editingText by mutableStateOf("")
        private set

    var isModelLoaded by mutableStateOf(false)
        private set

    var loadedModelName by mutableStateOf<String?>(null)
        private set

    var connectionState by mutableStateOf(ConnectionState.Unknown)
        private set

    var healthJson by mutableStateOf("")
        private set

    var gpuQueueDepth by mutableStateOf<Int?>(null)
        private set

    var errorBanner by mutableStateOf<String?>(null)
        private set

    var lastFailedPrompt by mutableStateOf<String?>(null)
        private set

    private val cpuMonitor = CpuMonitor()

    private val apiService: ApiService
        get() = ApiService.getInstance(currentApiType)

    private var lastTokenTime = System.currentTimeMillis()
    private var tokenCount = 0
    private var isFirstToken = true
    private var requestOutputTokens = 0
    private var requestFirstTokenAtMs = 0L
    private val pendingStream = StringBuilder()
    private var lastStreamFlushAt = 0L
    private var lastDatasetSpeedUiAt = 0L
    private val streamFlushIntervalMs = 64L

    var isTestRunning by mutableStateOf(false)
        private set

    var testTotal by mutableStateOf(0)
        private set

    var testProcessed by mutableStateOf(0)
        private set

    var datasetSpeed by mutableStateOf(0.0)
        private set

    var lastTestSpeed by mutableStateOf(0.0)
        private set

    private var testStartTimeMs: Long = 0L
    private var testTotalTokens: Int = 0
    private var testActiveTimeMs: Long = 0L
    private var testSkipped = 0
    private var testFailed = 0

    private var generateJob: Job? = null
    private var testJob: Job? = null
    private var streamingAssistantId: String? = null
    private var stoppedByUser = false
    private var generationSeq = 0

    private val httpClient = OkHttpClient.Builder()
        .callTimeout(5, TimeUnit.SECONDS)
        .connectTimeout(4, TimeUnit.SECONDS)
        .build()

    val canSend: Boolean
        get() = inferenceMode == InferenceMode.API || isModelLoaded

    init {
        llamaAndroid.setDraftStrategy(heteroDraftStrategy)
        llamaAndroid.setUseQueueInGoodput(false)
        viewModelScope.launch {
            while (isActive) {
                if (inferenceMode == InferenceMode.HETERO) {
                    pingHealthz()
                } else {
                    connectionState = when (inferenceMode) {
                        InferenceMode.API -> ConnectionState.Online
                        else -> ConnectionState.Unknown
                    }
                }
                delay(15_000)
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        cpuMonitor.stopMonitoring()
        testJob?.cancel()
        generateJob?.cancel()
        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                appendEval(exc.message ?: "unload failed")
            }
        }
    }

    fun updateCpuMonitorEnabled(enabled: Boolean) {
        if (enabled == isCpuMonitorEnabled) return
        isCpuMonitorEnabled = enabled
        if (enabled) {
            testOwnsMonitor = false
            startProcessCpuMonitor()
        } else if (!isTestRunning) {
            cpuMonitor.stopMonitoring()
            if (!hasProcessCpuSample) cpuUsage = -1.0f
        } else {
            testOwnsMonitor = true
        }
    }

    private fun startProcessCpuMonitor() {
        cpuMonitor.startMonitoring(500) { usage ->
            if (isTestRunning) {
                if (isInferenceRunning && kotlin.math.abs(cpuUsage - usage) >= 0.1f) {
                    cpuUsage = usage
                }
                return@startMonitoring
            }
            if (!isInferenceRunning) return@startMonitoring
            if (kotlin.math.abs(cpuUsage - usage) >= 0.1f || cpuUsage < 0f) {
                cpuUsage = usage
            }
        }
    }

    private fun beginInferenceMonitoring() {
        isInferenceRunning = true
        if (!isTestRunning) {
            hasProcessCpuSample = false
            cpuUsage = -1.0f
        }
    }

    private fun endInferenceMonitoring() {
        isInferenceRunning = false
    }

    fun send(skipAddingUserMessage: Boolean = false) {
        val seq = ++generationSeq
        generateJob?.cancel()
        generateJob = null

        val text = message
        message = ""
        errorBanner = null

        val lastReport = messages.lastOrNull { it.kind == MessageKind.Report }
        val isImageMessage = lastReport?.image != null

        if (!isImageMessage && !skipAddingUserMessage && text.isNotBlank()) {
            messages += ChatMessage.report(text)
        }

        val assistant = ChatMessage.assistant("")
        streamingAssistantId = assistant.id
        messages += assistant

        lastTokenTime = System.currentTimeMillis()
        tokenCount = 0
        inferenceSpeed = 0.0
        isFirstToken = true
        requestOutputTokens = 0
        requestFirstTokenAtMs = 0L
        requestTpotMs = -1.0
        pendingStream.setLength(0)
        lastStreamFlushAt = 0L
        stoppedByUser = false
        lastFailedPrompt = text

        beginInferenceMonitoring()
        generateJob = viewModelScope.launch {
            try {
                val fullPrompt = buildPrompt(text)
                val flow = when (inferenceMode) {
                    InferenceMode.LOCAL -> llamaAndroid.send(fullPrompt, formatChat = usesSpecialChatTokens(promptMode))
                    InferenceMode.API -> apiService.send(fullPrompt)
                    InferenceMode.HETERO -> llamaAndroid.sendHetero(
                        fullPrompt,
                        formatChat = usesSpecialChatTokens(promptMode),
                        serverUrl = heteroServerUrl,
                    )
                }
                flow.catch { err ->
                    Log.e(tag, "send() failed", err)
                    appendEval(err.message ?: "send failed")
                    errorBanner = err.message ?: "解读失败"
                }.collect { (str, tokens) ->
                    lastFailedPrompt = null
                    updateMessageAndSpeed(str, tokens)
                }
            } catch (e: CancellationException) {
                if (!stoppedByUser && seq == generationSeq) {
                    markStopped()
                }
                throw e
            } catch (e: Exception) {
                Log.e(tag, "Error during inference", e)
                appendEval("Error: ${e.message}")
                errorBanner = e.message ?: "解读失败"
            } finally {
                if (seq == generationSeq) {
                    flushStreamingText()
                    endInferenceMonitoring()
                    generateJob = null
                }
            }
        }
    }

    fun stopGeneration() {
        val stoppingTest = isTestRunning
        if (generateJob == null && !isInferenceRunning && !stoppingTest && testJob?.isActive != true) return
        stoppedByUser = true
        generationSeq++
        generateJob?.cancel()
        generateJob = null
        if (stoppingTest) {
            isTestRunning = false
            publishTestStatus("测试已停止：$testProcessed/$testTotal")
        }
        testJob?.cancel()
        testJob = null
        flushStreamingText()
        markStopped()
        endInferenceMonitoring()
    }

    fun retryLast() {
        val prompt = lastFailedPrompt ?: return
        errorBanner = null
        message = prompt
        send(skipAddingUserMessage = true)
    }

    fun dismissError() {
        errorBanner = null
    }

    private fun markStopped() {
        val id = streamingAssistantId ?: return
        messages = messages.map { msg ->
            if (msg.id != id) {
                msg
            } else {
                val text = msg.content.trim()
                msg.copy(content = if (text.isEmpty()) "已停止" else "$text\n\n（已停止）")
            }
        }
    }

    private fun buildPrompt(text: String): String = when (promptMode) {
        PromptMode.QWEN2 ->
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:"
        PromptMode.QWEN3 ->
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nUser: $text\n\nAssistant:\n<think></think>"
        PromptMode.QWEN35 ->
            if (inferenceMode == InferenceMode.API && currentApiType == ApiType.QWEN38) {
                // Qwen3.8 thinking mode is the model-card default: open <think> and do not close it.
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$text<|im_end|>\n<|im_start|>assistant\n<think>\n"
            } else {
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$text<|im_end|>\n<|im_start|>assistant\n"
            }
        PromptMode.MINICPM5 ->
            // Official MiniCPM5 jinja: BOS + ChatML, then empty think (enable_thinking=false).
            // GGUF has add_bos_token=false, so BOS must be in the prompt. Missing either
            // BOS or the think block makes greedy decoding collapse to repeating `**`.
            "<s><|im_start|>user\n$text<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        PromptMode.DEEPSEEk -> text
    }

    private fun updateMessageAndSpeed(str: String, tokens: Int) {
        if (str.isNotEmpty()) pendingStream.append(str)
        if (tokens > 0) tokenCount += tokens
        val currentTime = System.currentTimeMillis()
        if (requestFirstTokenAtMs == 0L && (str.isNotEmpty() || tokens > 0)) {
            requestFirstTokenAtMs = currentTime
        }
        if (tokens > 0) {
            requestOutputTokens += tokens
            val produced = requestOutputTokens - 1
            val elapsed = currentTime - requestFirstTokenAtMs
            if (produced > 0 && elapsed > 0L) {
                requestTpotMs = elapsed.toDouble() / produced
            }
        }
        val timeDiff = (currentTime - lastTokenTime) / 1000.0

        if (isFirstToken) {
            lastTokenTime = currentTime
            isFirstToken = false
        } else if (timeDiff >= 1.0) {
            val speed = tokenCount / timeDiff
            if (kotlin.math.abs(speed - inferenceSpeed) >= 0.02) {
                inferenceSpeed = speed
            }
            tokenCount = 0
            lastTokenTime = currentTime
        }
        if (currentTime - lastStreamFlushAt >= streamFlushIntervalMs) {
            flushStreamingText()
        }
    }

    private fun flushStreamingText() {
        if (pendingStream.isEmpty()) return
        val id = streamingAssistantId ?: messages.lastOrNull { it.kind == MessageKind.Assistant }?.id
        if (id == null) {
            pendingStream.setLength(0)
            return
        }
        val chunk = pendingStream.toString()
        pendingStream.setLength(0)
        lastStreamFlushAt = System.currentTimeMillis()
        val idx = messages.indexOfFirst { it.id == id }
        if (idx < 0) return
        val next = messages.toMutableList()
        next[idx] = next[idx].copy(content = next[idx].content + chunk)
        messages = next
    }

    fun isSelectedModel(model: Downloadable): Boolean {
        if (model.isApiModel) {
            if (inferenceMode != InferenceMode.API) return false
            return model.name == selectedApiName
        }
        if (!isModelLoaded) return false
        if (model.destination?.name != loadedModelName) return false
        return model.isHetero == (inferenceMode == InferenceMode.HETERO)
    }

    fun load(pathToModel: String, isHetero: Boolean = false) {
        viewModelScope.launch {
            try {
                val file = File(pathToModel)
                if (!file.isCompleteGguf(advertisedMinBytes(pathToModel, file.name))) {
                    val size = if (file.exists()) file.length() else 0L
                    appendEval("模型文件不完整（$size 字节），已删除，请重新下载")
                    file.delete()
                    isModelLoaded = false
                    loadedModelName = null
                    return@launch
                }
                inferenceMode = if (isHetero) InferenceMode.HETERO else InferenceMode.LOCAL
                try {
                    llamaAndroid.unload()
                } catch (_: IllegalStateException) {
                }
                isModelLoaded = false
                llamaAndroid.load(pathToModel)
                isModelLoaded = true
                val fileName = pathToModel.substringAfterLast("/")
                loadedModelName = fileName
                promptMode = promptModeForModel(fileName)
                if (isHetero) {
                    llamaAndroid.setHeteroIcnIdentity(
                        heteroIcnTarget(fileName),
                        heteroIcnTokenizer(fileName),
                    )
                    appendEval("使用草稿模型：$fileName")
                    if (fileName.contains("Qwen3.5") || fileName.contains("Qwen3.8")) {
                        appendEval("云端应对齐 Qwen3.8-27B（纯文本，不传图像）")
                    }
                    pingHealthz()
                } else {
                    appendEval("已切换到模型：$fileName")
                }
            } catch (exc: CancellationException) {
                throw exc
            } catch (exc: Exception) {
                Log.e(tag, "load() failed", exc)
                isModelLoaded = false
                loadedModelName = null
                appendEval("加载失败：${exc.message ?: exc.javaClass.simpleName}。若刚中断过下载，请重新下载。")
            }
        }
    }

    fun updateMessage(newMessage: String) {
        message = newMessage
    }

    fun clear() {
        generateJob?.cancel()
        messages = listOf(ChatMessage.evalLog("已清除对话历史"))
        inferenceSpeed = 0.0
        requestTpotMs = -1.0
        requestOutputTokens = 0
        requestFirstTokenAtMs = 0L
        errorBanner = null
        streamingAssistantId = null
    }

    fun log(message: String) {
        appendEval(message)
    }

    private fun appendEval(text: String) {
        messages += ChatMessage.evalLog(text)
    }

    fun switchToApiMode(type: ApiType, name: String) {
        currentApiType = type
        selectedApiName = name
        inferenceMode = InferenceMode.API
        connectionState = ConnectionState.Online
        if (type == ApiType.QWEN38) {
            promptMode = PromptMode.QWEN35
            QwenApiService.getInstance().configure(heteroServerUrl, qwen38 = true)
            appendEval("已切换到 $name，思考模式 temperature=1.0 top_p=0.95 top_k=20")
        } else if (type == ApiType.QWEN) {
            QwenApiService.getInstance().configure(heteroServerUrl, qwen38 = false)
            appendEval("已切换到 $name，请求发到 HeteroSpec /generate，贪心解码。云端权重是服务器当前加载的模型")
        } else {
            appendEval("已切换到 $name")
        }
        loadedModelName = name
    }

    fun updateHeteroServerUrl(url: String) {
        heteroServerUrl = url.trim().ifEmpty { LLamaAndroid.DEFAULT_HETERO_SERVER_URL }
        llamaAndroid.setHeteroServerUrl(heteroServerUrl)
        if (currentApiType == ApiType.QWEN || currentApiType == ApiType.QWEN38) {
            QwenApiService.getInstance().configure(
                heteroServerUrl,
                qwen38 = currentApiType == ApiType.QWEN38,
            )
        }
        appendEval("HeteroSpec服务地址：$heteroServerUrl")
        if (inferenceMode == InferenceMode.HETERO) {
            viewModelScope.launch { pingHealthz() }
        }
    }

    fun updateHeteroDraftStrategy(strategy: String) {
        llamaAndroid.setDraftStrategy(strategy)
        heteroDraftStrategy = llamaAndroid.getDraftStrategy()
    }

    fun updateUseQueueInGoodput(enabled: Boolean) {
        llamaAndroid.setUseQueueInGoodput(enabled)
    }

    fun addImageMessage(bitmap: Bitmap) {
        messages += ChatMessage.report("", bitmap)
    }

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
        if (!canSend) {
            errorBanner = "请先到设置里选择模型"
            hideOcrEditor()
            return
        }
        val ocrMessage = "请解读病例报告并给出简短建议：$ocrEditText"
        val reportIndex = messages.indexOfLast { it.kind == MessageKind.Report }
        if (reportIndex >= 0) {
            val old = messages[reportIndex]
            messages = messages.toMutableList().also {
                it[reportIndex] = old.copy(content = ocrEditText)
            }
        } else {
            messages += ChatMessage.report(ocrEditText)
        }
        message = ocrMessage
        hideOcrEditor()
        send(skipAddingUserMessage = true)
        message = ""
    }

    fun startEditingMessage(id: String, content: String) {
        editingMessageId = id
        editingText = content
    }

    fun cancelEditingMessage() {
        editingMessageId = null
        editingText = ""
    }

    fun updateEditingText(text: String) {
        editingText = text
    }

    fun confirmEditMessage() {
        val text = editingText
        if (editingMessageId != null && text.isNotBlank()) {
            messages += ChatMessage.report(text)
            message = text
            cancelEditingMessage()
            send(skipAddingUserMessage = true)
            message = ""
        }
    }

    fun prepareDatasetTest(): String? {
        if (isTestRunning || testJob?.isActive == true) {
            val reason = "上一次测试还没结束"
            reportTestProblem(reason)
            return reason
        }
        val reason = testBlockReason() ?: return null
        reportTestProblem(reason)
        return reason
    }

    fun attachTestJob(job: Job) {
        testJob = job
    }

    fun onTestCancelled() {
        if (!isTestRunning) return
        isTestRunning = false
        endInferenceMonitoring()
        if (testOwnsMonitor && !isCpuMonitorEnabled) {
            testOwnsMonitor = false
            cpuMonitor.stopMonitoring()
        }
        publishTestStatus("测试已停止：$testProcessed/$testTotal")
    }

    fun testBlockReason(): String? = when (inferenceMode) {
        InferenceMode.LOCAL, InferenceMode.HETERO ->
            if (isModelLoaded) {
                null
            } else {
                "请先在设置里加载模型，再开始测试"
            }
        InferenceMode.API -> when (currentApiType) {
            ApiType.DEEPSEEK ->
                if (DeepseekApiService.getInstance().isReady()) {
                    null
                } else {
                    "DeepSeek API 还没有可用的密钥，这条在线服务现在发不出请求"
                }
            ApiType.QWEN, ApiType.QWEN38 ->
                if (QwenApiService.getInstance().hasEndpoint()) {
                    null
                } else {
                    "在线服务地址是空的，请先在设置里保存 HeteroSpec 地址"
                }
        }
    }

    fun startTest(totalItems: Int, strategyLabel: String? = null) {
        isTestRunning = true
        testTotal = totalItems
        testProcessed = 0
        datasetSpeed = 0.0
        lastTestSpeed = 0.0
        testTotalTokens = 0
        testStartTimeMs = SystemClock.elapsedRealtime()
        testActiveTimeMs = 0L
        lastDatasetSpeedUiAt = 0L
        testSkipped = 0
        testFailed = 0
        datasetCpuMs = 0L
        datasetWallMs = 0L
        datasetCpuAvg = -1.0f
        hasProcessCpuSample = false
        cpuUsage = -1.0f
        if (!isCpuMonitorEnabled) {
            testOwnsMonitor = true
            startProcessCpuMonitor()
        }
        val model = loadedModelName ?: "未命名"
        val extra = if (strategyLabel.isNullOrEmpty()) "" else "，策略=$strategyLabel"
        val text = "开始测试：$model，$inferenceMode，共 $totalItems 条$extra"
        publishTestStatus(text)
        if (inferenceMode == InferenceMode.API) {
            appendEval("在线测试每条最多 $TEST_API_MAX_NEW_TOKENS 个新 token")
        }
    }

    fun skipTestItem() {
        if (!isTestRunning) return
        testSkipped += 1
        testProcessed += 1
    }

    private fun updateTestTokens(tokens: Int) {
        if (!isTestRunning) return
        if (tokens <= 0) return
        testTotalTokens += tokens
    }

    fun finishTest(strategyLabel: String? = null) {
        if (!isTestRunning) return
        val activeSeconds = testActiveTimeMs / 1000.0
        lastTestSpeed = if (activeSeconds > 0) testTotalTokens / activeSeconds else 0.0
        datasetSpeed = lastTestSpeed
        isTestRunning = false
        if (datasetWallMs > 0L) {
            datasetCpuAvg = processCpuPercent(datasetCpuMs, datasetWallMs, cpuCores) ?: -1.0f
            cpuUsage = datasetCpuAvg
            hasProcessCpuSample = datasetCpuAvg >= 0f
        }
        if (testOwnsMonitor && !isCpuMonitorEnabled) {
            testOwnsMonitor = false
            cpuMonitor.stopMonitoring()
        }
        val extra = if (strategyLabel.isNullOrEmpty()) "" else "，策略=$strategyLabel"
        val skipped = if (testSkipped > 0) "，跳过 $testSkipped 条空识别" else ""
        val failed = if (testFailed > 0) "，失败 $testFailed 条" else ""
        val cpuText = if (datasetWallMs > 0L && datasetCpuAvg >= 0f) {
            val coresUsed = datasetCpuMs.toDouble() / datasetWallMs.toDouble()
            "，进程 CPU ${"%.1f".format(datasetCpuAvg)}%（${"%.2f".format(coresUsed)}/$cpuCores 核）"
        } else {
            ""
        }
        publishTestStatus(
            "测试完成：$testProcessed/$testTotal$extra$skipped$failed，" +
                "整体速度 ${"%.2f".format(lastTestSpeed)} tokens/s（含等待）$cpuText"
        )
    }

    private fun publishTestSpeed(nowMs: Long) {
        val activeSeconds = testActiveTimeMs / 1000.0
        if (activeSeconds <= 0.0) return
        if (nowMs - lastDatasetSpeedUiAt < 200L) return
        lastDatasetSpeedUiAt = nowMs
        datasetSpeed = testTotalTokens / activeSeconds
    }

    suspend fun sendForTest(text: String) {
        sendForTest_impl(text)
    }

    suspend fun sendForTest_impl(text: String) {
        if (text.isBlank()) {
            skipTestItem()
            return
        }
        beginInferenceMonitoring()
        val wallStart = SystemClock.elapsedRealtime()
        val cpuStart = Process.getElapsedCpuTime()
        var produced = 0
        try {
            val fullPrompt = buildPrompt(text)
            val flow = when (inferenceMode) {
                InferenceMode.LOCAL -> llamaAndroid.send(
                    fullPrompt,
                    formatChat = usesSpecialChatTokens(promptMode),
                )
                InferenceMode.API -> when (val api = apiService) {
                    is QwenApiService -> api.send(fullPrompt, TEST_API_MAX_NEW_TOKENS)
                    else -> api.send(fullPrompt)
                }
                InferenceMode.HETERO -> llamaAndroid.sendHetero(
                    fullPrompt,
                    formatChat = usesSpecialChatTokens(promptMode),
                    serverUrl = heteroServerUrl,
                )
            }
            flow.collect { (_, tokens) ->
                if (!isTestRunning) return@collect
                if (tokens > 0) {
                    produced += tokens
                    updateTestTokens(tokens)
                }
                val partialWall = SystemClock.elapsedRealtime() - wallStart
                val countedWall = testActiveTimeMs + if (produced > 0) partialWall else 0L
                if (countedWall > 0L) {
                    val now = SystemClock.elapsedRealtime()
                    val previous = testActiveTimeMs
                    testActiveTimeMs = countedWall
                    publishTestSpeed(now)
                    testActiveTimeMs = previous
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(tag, "Error during sendForTest", e)
            testFailed += 1
            appendEval("这一条失败：${e.message}")
        } finally {
            endInferenceMonitoring()
            if (isTestRunning) {
                val wall = SystemClock.elapsedRealtime() - wallStart
                val cpu = (Process.getElapsedCpuTime() - cpuStart).coerceAtLeast(0L)
                if (produced > 0 && wall > 0L) {
                    testActiveTimeMs += wall
                    datasetCpuMs += cpu
                    datasetWallMs += wall
                    val average = processCpuPercent(datasetCpuMs, datasetWallMs, cpuCores)
                    if (average != null) {
                        datasetCpuAvg = average
                        cpuUsage = average
                    }
                    publishTestSpeed(SystemClock.elapsedRealtime())
                }
                testProcessed += 1
            }
        }
    }

    private fun publishTestStatus(text: String) {
        messages += ChatMessage.status(text)
        appendEval(text)
    }

    private fun reportTestProblem(text: String) {
        errorBanner = text
        publishTestStatus(text)
    }

    private suspend fun pingHealthz() {
        val url = healthzUrlFromWs(heteroServerUrl)
        if (url.isEmpty()) {
            connectionState = ConnectionState.Unknown
            return
        }
        try {
            val body = withContext(Dispatchers.IO) {
                val request = Request.Builder().url(url).get().build()
                httpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        throw IOException("HTTP ${response.code}")
                    }
                    response.body?.string().orEmpty()
                }
            }
            healthJson = body
            val json = JSONObject(body)
            gpuQueueDepth = json.optInt("gpu_queue_depth", -1).takeIf { it >= 0 }
            val targetModel = json.optString("target_model").ifBlank { json.optString("model") }
            val tokenizerId = json.optString("tokenizer_id")
            if (targetModel.isNotBlank()) {
                llamaAndroid.setHeteroIcnIdentity(
                    targetModel,
                    tokenizerId.ifBlank { "$targetModel-tokenizer" },
                )
            }
            val loaded = loadedModelName.orEmpty()
            if ((loaded.contains("Qwen3.5") || loaded.contains("Qwen3.8")) &&
                !targetModel.contains("3.8") &&
                !targetModel.contains("3.5")
            ) {
                appendEval("云端当前是 $targetModel，这个草稿需要 Qwen3.8-27B（248320 词表）")
            }
            connectionState = ConnectionState.Online
        } catch (e: Exception) {
            connectionState = ConnectionState.Offline
            healthJson = e.message ?: "offline"
            gpuQueueDepth = null
        }
    }
}
