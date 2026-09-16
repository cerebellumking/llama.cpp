package com.example.llama.ui.eval

import android.app.DownloadManager
import android.content.Context
import android.llama.cpp.LLamaAndroid
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.example.llama.Downloadable
import com.example.llama.MainViewModel
import com.example.llama.api.ApiType
import com.example.llama.promptModeForModel
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EvalSettingsSheet(
    viewModel: MainViewModel,
    models: List<Downloadable>,
    dm: DownloadManager,
    evalUnlocked: Boolean,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val preferences = remember {
        context.getSharedPreferences("heterospec_settings", Context.MODE_PRIVATE)
    }
    var heteroServerInput by remember { mutableStateOf(viewModel.heteroServerUrl) }
    var currentModelName by remember { mutableStateOf(viewModel.loadedModelName ?: "") }
    var downloadingModel by remember { mutableStateOf<Downloadable?>(null) }
    var downloadProgress by remember { mutableDoubleStateOf(0.0) }
    var downloadId by remember { mutableLongStateOf(-1L) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    LaunchedEffect(downloadId) {
        if (downloadId == -1L) return@LaunchedEffect
        while (true) {
            val cursor = dm.query(DownloadManager.Query().setFilterById(downloadId))
            if (cursor != null && cursor.moveToFirst()) {
                val bytesDownloaded = cursor.getLong(
                    cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                )
                val bytesTotal = cursor.getLong(
                    cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                )
                if (bytesTotal > 0) {
                    downloadProgress = bytesDownloaded.toDouble() / bytesTotal
                }
                if (bytesDownloaded == bytesTotal && bytesTotal > 0) {
                    downloadingModel?.let { model ->
                        viewModel.clear()
                        viewModel.promptMode = promptModeForModel(model.name)
                        viewModel.load(model.destination!!.path, model.isHetero)
                        currentModelName = model.name
                    }
                    downloadingModel = null
                    downloadId = -1L
                    downloadProgress = 0.0
                    cursor.close()
                    break
                }
            }
            cursor?.close()
            delay(1000)
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        modifier = modifier,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 8.dp),
        ) {
            Text("设置", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(12.dp))
            OutlinedTextField(
                value = heteroServerInput,
                onValueChange = { heteroServerInput = it },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                label = { Text("云端服务地址") },
                placeholder = { Text(LLamaAndroid.DEFAULT_HETERO_SERVER_URL) },
            )
            TextButton(
                onClick = {
                    val nextUrl = heteroServerInput.trim().ifEmpty {
                        LLamaAndroid.DEFAULT_HETERO_SERVER_URL
                    }
                    preferences.edit().putString("heterospec_server_url", nextUrl).apply()
                    viewModel.updateHeteroServerUrl(nextUrl)
                    heteroServerInput = nextUrl
                }
            ) {
                Text("保存地址")
            }
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            Text("选择模型", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            ModelGroupHeader("在线服务")
            models.filter { it.isApiModel }.forEach { model ->
                DropdownMenuItem(
                    text = { Text(model.name) },
                    onClick = {
                        viewModel.clear()
                        viewModel.promptMode = promptModeForModel(model.name)
                        viewModel.switchToApiMode(
                            when (model.name) {
                                "DeepSeek API" -> ApiType.DEEPSEEK
                                else -> ApiType.QWEN
                            }
                        )
                        currentModelName = model.name
                    },
                )
            }
            ModelGroupHeader("仅用手机")
            models.filter { !it.isApiModel && !it.isHetero }.forEach { model ->
                ModelRow(
                    model = model,
                    downloadingModel = downloadingModel,
                    downloadProgress = downloadProgress,
                    currentModelName = currentModelName,
                    onLoadOrDownload = { selected ->
                        if (selected.destination?.exists() == true) {
                            viewModel.clear()
                            viewModel.promptMode = promptModeForModel(selected.name)
                            viewModel.load(selected.destination.path)
                            currentModelName = selected.name
                        } else if (downloadingModel == null && selected.source != null) {
                            val request = DownloadManager.Request(selected.source)
                                .setTitle(selected.name)
                                .setDescription("正在下载模型...")
                                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                                .setDestinationUri(selected.destination!!.toUri())
                            downloadId = dm.enqueue(request)
                            downloadingModel = selected
                            viewModel.log("开始下载模型：${selected.name}")
                        }
                    },
                )
            }
            ModelGroupHeader("云端协同")
            models.filter { it.isHetero }.forEach { model ->
                ModelRow(
                    model = model,
                    downloadingModel = downloadingModel,
                    downloadProgress = downloadProgress,
                    currentModelName = currentModelName,
                    onLoadOrDownload = { selected ->
                        if (selected.destination?.exists() == true) {
                            viewModel.clear()
                            viewModel.promptMode = promptModeForModel(selected.name)
                            viewModel.load(selected.destination.path, true)
                            currentModelName = selected.name
                        } else if (downloadingModel == null && selected.source != null) {
                            val request = DownloadManager.Request(selected.source)
                                .setTitle(selected.name)
                                .setDescription("正在下载模型...")
                                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                                .setDestinationUri(selected.destination!!.toUri())
                            downloadId = dm.enqueue(request)
                            downloadingModel = selected
                            viewModel.log("开始下载草稿模型：${selected.name}")
                        }
                    },
                )
            }
            if (evalUnlocked) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                Text("验收", style = MaterialTheme.typography.titleMedium)
                Text(
                    text = "healthz\n${viewModel.healthJson.ifBlank { "（尚无探测结果）" }}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                TextButton(onClick = { viewModel.clear() }) { Text("清除对话") }
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun ModelGroupHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 8.dp, bottom = 4.dp),
    )
}

@Composable
private fun ModelRow(
    model: Downloadable,
    downloadingModel: Downloadable?,
    downloadProgress: Double,
    currentModelName: String,
    onLoadOrDownload: (Downloadable) -> Unit,
) {
    DropdownMenuItem(
        text = {
            Column {
                Text(model.name)
                if (downloadingModel == model) {
                    LinearProgressIndicator(
                        progress = { downloadProgress.toFloat() },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                    )
                    Text("${(downloadProgress * 100).toInt()}%", style = MaterialTheme.typography.labelSmall)
                } else if (currentModelName == model.name) {
                    Text("当前", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                }
            }
        },
        onClick = { onLoadOrDownload(model) },
        enabled = downloadingModel == null || downloadingModel == model,
    )
}
