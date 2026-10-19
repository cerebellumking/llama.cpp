package com.example.llama.ui.workspace

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable
import com.example.llama.ChatMessage
import com.example.llama.ConnectionState
import com.example.llama.InferenceMode

@Immutable
data class EvalMetrics(
    val inferenceSpeed: Double = 0.0,
    val requestTpotMs: Double = -1.0,
    val isInferenceRunning: Boolean = false,
    val cpuUsage: Float = -1.0f,
    val cpuEnabled: Boolean = false,
    val cpuMeasured: Boolean = false,
    val datasetSpeed: Double = 0.0,
    val lastTestSpeed: Double = 0.0,
    val testRunning: Boolean = false,
    val testProcessed: Int = 0,
    val testTotal: Int = 0,
    val draftStrategy: String = "",
    val heteroServerUrl: String = "",
    val healthJson: String = "",
    val gpuQueueDepth: Int? = null,
    val logs: List<ChatMessage> = emptyList(),
)

@Immutable
data class WorkspaceUi(
    val feed: List<ChatMessage> = emptyList(),
    val input: String = "",
    val isInferenceRunning: Boolean = false,
    val canSend: Boolean = false,
    val isShowingOcr: Boolean = false,
    val ocrText: String = "",
    val ocrImage: Bitmap? = null,
    val editingMessageId: String? = null,
    val editingText: String = "",
    val connection: ConnectionState = ConnectionState.Unknown,
    val inferenceMode: InferenceMode = InferenceMode.LOCAL,
    val loadedModelName: String? = null,
    val errorBanner: String? = null,
    val eval: EvalMetrics = EvalMetrics(),
)

data class WorkspaceEvents(
    val onInputChange: (String) -> Unit = {},
    val onSend: () -> Unit = {},
    val onStop: () -> Unit = {},
    val onCamera: () -> Unit = {},
    val onPickImage: () -> Unit = {},
    val onEditMessage: (String, String) -> Unit = { _, _ -> },
    val onUpdateEditingText: (String) -> Unit = {},
    val onConfirmEdit: () -> Unit = {},
    val onCancelEdit: () -> Unit = {},
    val onOcrTextChange: (String) -> Unit = {},
    val onConfirmOcr: () -> Unit = {},
    val onCancelOcr: () -> Unit = {},
    val onRetry: () -> Unit = {},
    val onDismissError: () -> Unit = {},
    val onTitleTap: () -> Unit = {},
    val onOpenSettings: () -> Unit = {},
    val onOpenTest: () -> Unit = {},
    val onToggleCpu: (Boolean) -> Unit = {},
)
