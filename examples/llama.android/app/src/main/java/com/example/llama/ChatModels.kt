package com.example.llama

import androidx.compose.runtime.Immutable
import android.graphics.Bitmap
import java.util.UUID

enum class MessageType {
    USER,
    SYSTEM,
}

enum class MessageKind {
    Report,
    Assistant,
    Status,
    EvalLog,
}

enum class InferenceMode {
    LOCAL,
    API,
    HETERO,
}

enum class PromptMode {
    QWEN2,
    DEEPSEEk,
    QWEN3,
}

enum class ConnectionState {
    Unknown,
    Online,
    Offline,
}

@Immutable
data class ChatMessage(
    val content: String,
    val type: MessageType,
    val image: Bitmap? = null,
    val id: String = UUID.randomUUID().toString(),
    val kind: MessageKind = defaultKind(type),
) {
    companion object {
        fun report(content: String, image: Bitmap? = null): ChatMessage =
            ChatMessage(content, MessageType.USER, image, kind = MessageKind.Report)

        fun assistant(content: String, id: String = UUID.randomUUID().toString()): ChatMessage =
            ChatMessage(content, MessageType.SYSTEM, id = id, kind = MessageKind.Assistant)

        fun evalLog(content: String): ChatMessage =
            ChatMessage(content, MessageType.SYSTEM, kind = MessageKind.EvalLog)

        fun status(content: String): ChatMessage =
            ChatMessage(content, MessageType.SYSTEM, kind = MessageKind.Status)

        private fun defaultKind(type: MessageType): MessageKind = when (type) {
            MessageType.USER -> MessageKind.Report
            MessageType.SYSTEM -> MessageKind.Assistant
        }
    }
}

fun promptModeForModel(modelName: String): PromptMode = when {
    modelName.contains("DeepSeek API") -> PromptMode.DEEPSEEk
    modelName.contains("Qwen2.5") -> PromptMode.QWEN2
    modelName.contains("DeepSeek-R1-DRAFT") -> PromptMode.QWEN2
    modelName.contains("Qwen3") -> PromptMode.QWEN3
    else -> PromptMode.QWEN2
}

fun healthzUrlFromWs(wsUrl: String): String {
    val trimmed = wsUrl.trim()
    if (trimmed.isEmpty()) return ""
    val uri = android.net.Uri.parse(trimmed)
    val scheme = when (uri.scheme?.lowercase()) {
        "wss", "https" -> "https"
        else -> "http"
    }
    val host = uri.host ?: return ""
    val port = uri.port
    val authority = if (port != -1) "$host:$port" else host
    return "$scheme://$authority/healthz"
}
