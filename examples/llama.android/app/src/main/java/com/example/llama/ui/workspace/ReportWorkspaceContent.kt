package com.example.llama.ui.workspace

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.llama.ChatMessage
import com.example.llama.ConnectionState
import com.example.llama.InferenceMode
import com.example.llama.MessageKind
import com.example.llama.ui.theme.LlamaAndroidTheme
import kotlinx.coroutines.delay

@Composable
fun ReportWorkspaceContent(
    ui: WorkspaceUi,
    events: WorkspaceEvents,
    modifier: Modifier = Modifier,
) {
    val scheme = MaterialTheme.colorScheme
    val listState = rememberLazyListState()
    val lastId = ui.feed.lastOrNull()?.id

    LaunchedEffect(lastId) {
        if (ui.feed.isNotEmpty()) {
            listState.scrollToItem(ui.feed.lastIndex, 50_000)
        }
    }
    LaunchedEffect(ui.isInferenceRunning, lastId) {
        if (!ui.isInferenceRunning) return@LaunchedEffect
        while (true) {
            delay(120)
            val lastIndex = listState.layoutInfo.totalItemsCount - 1
            if (lastIndex < 0) continue
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: -1
            if (lastVisible >= lastIndex - 1) {
                listState.scrollToItem(lastIndex, 50_000)
            }
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(scheme.background),
    ) {
        if (ui.connection == ConnectionState.Offline && ui.inferenceMode == InferenceMode.HETERO) {
            StatusStrip(text = "云端未连接，可在设置中修改地址或改用本机模型。", error = true)
        }
        if (ui.isInferenceRunning) {
            val waiting = ui.feed.lastOrNull { it.kind == MessageKind.Assistant }?.content.isNullOrBlank()
            StatusStrip(text = if (waiting) "正在对接云端…" else "正在解读…")
        }
        ui.errorBanner?.let { banner ->
            ErrorStrip(text = banner, onRetry = events.onRetry, onDismiss = events.onDismissError)
        }
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
        ) {
            if (ui.feed.isEmpty()) {
                EmptyWorkspace(canSend = ui.canSend)
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    items(
                        items = ui.feed,
                        key = { it.id },
                        contentType = { it.kind },
                    ) { msg ->
                        val onEdit = remember(msg.id, msg.content, events) {
                            { events.onEditMessage(msg.id, msg.content) }
                        }
                        when (msg.kind) {
                            MessageKind.Report -> ReportCard(
                                content = msg.content,
                                image = msg.image,
                                isEditing = ui.editingMessageId == msg.id,
                                editingText = if (ui.editingMessageId == msg.id) ui.editingText else "",
                                onEdit = onEdit,
                                onUpdateEditingText = events.onUpdateEditingText,
                                onConfirmEdit = events.onConfirmEdit,
                                onCancelEdit = events.onCancelEdit,
                            )
                            MessageKind.Assistant -> ReadingPane(
                                content = msg.content,
                                streaming = ui.isInferenceRunning && msg.id == lastId,
                            )
                            MessageKind.Status -> StatusStrip(text = msg.content)
                            MessageKind.EvalLog -> {}
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusStrip(text: String, error: Boolean = false) {
    val scheme = MaterialTheme.colorScheme
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(if (error) scheme.error.copy(alpha = 0.12f) else scheme.primaryContainer)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = if (error) scheme.error else scheme.onPrimaryContainer,
        )
    }
}

@Composable
private fun ErrorStrip(text: String, onRetry: () -> Unit, onDismiss: () -> Unit) {
    val scheme = MaterialTheme.colorScheme
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(scheme.error.copy(alpha = 0.12f))
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            Icons.Default.ErrorOutline,
            contentDescription = null,
            tint = scheme.error,
            modifier = Modifier.size(18.dp),
        )
        Spacer(Modifier.size(8.dp))
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = scheme.error,
            modifier = Modifier.weight(1f),
            maxLines = 2,
        )
        TextButton(onClick = onRetry) { Text("点此续传") }
        TextButton(onClick = onDismiss) { Text("关闭") }
    }
}

@Composable
private fun EmptyWorkspace(canSend: Boolean) {
    val scheme = MaterialTheme.colorScheme
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Default.PhotoCamera,
            contentDescription = null,
            tint = scheme.primary,
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(14.dp)),
        )
        Spacer(Modifier.size(16.dp))
        Text(
            text = "拍一张检验单开始解读",
            style = MaterialTheme.typography.titleMedium,
            color = scheme.onSurface,
        )
        Spacer(Modifier.size(8.dp))
        Text(
            text = if (canSend) {
                "拍摄或从相册选择报告，核对文字后即可生成解读。"
            } else {
                "请先点右上角设置，选择要使用的模型。"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = scheme.onSurfaceVariant,
        )
    }
}

@Preview(showBackground = true, name = "empty")
@Composable
private fun PreviewEmpty() {
    LlamaAndroidTheme {
        ReportWorkspaceContent(
            ui = WorkspaceUi(),
            events = WorkspaceEvents(),
        )
    }
}

@Preview(showBackground = true, name = "reading")
@Composable
private fun PreviewReading() {
    LlamaAndroidTheme {
        ReportWorkspaceContent(
            ui = WorkspaceUi(
                feed = listOf(
                    ChatMessage.report("白细胞 6.2×10^9/L，血红蛋白 132 g/L"),
                    ChatMessage.assistant("血常规大致正常，建议结合临床症状随访。"),
                ),
                canSend = true,
                inferenceMode = InferenceMode.HETERO,
                connection = ConnectionState.Online,
            ),
            events = WorkspaceEvents(),
        )
    }
}

@Preview(showBackground = true, name = "eval")
@Composable
private fun PreviewEval() {
    LlamaAndroidTheme {
        ReportWorkspaceContent(
            ui = WorkspaceUi(
                canSend = true,
                isInferenceRunning = true,
                errorBanner = "云端暂时不可用",
            ),
            events = WorkspaceEvents(),
        )
    }
}
