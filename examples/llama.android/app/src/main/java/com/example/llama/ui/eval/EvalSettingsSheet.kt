package com.example.llama.ui.eval

import android.app.DownloadManager
import android.content.Context
import android.content.SharedPreferences
import android.llama.cpp.LLamaAndroid
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import androidx.compose.material3.RadioButton
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
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.example.llama.CpuMeterMode
import com.example.llama.Downloadable
import com.example.llama.MainViewModel
import com.example.llama.cancelDownloadsFor
import com.example.llama.ocr.DualOcrController
import com.example.llama.ocr.OcrBackend
import com.example.llama.api.ApiType
import com.example.llama.promptModeForModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EvalSettingsSheet(
    viewModel: MainViewModel,
    models: List<Downloadable>,
    dm: DownloadManager,
    evalUnlocked: Boolean,
    ocr: DualOcrController,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val preferences = remember {
        context.getSharedPreferences("heterospec_settings", Context.MODE_PRIVATE)
    }
    var heteroServerInput by remember { mutableStateOf(viewModel.heteroServerUrl) }
    var downloadingModel by remember { mutableStateOf<Downloadable?>(null) }
    var downloadProgress by remember { mutableDoubleStateOf(0.0) }
    var downloadId by remember { mutableLongStateOf(-1L) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScope()
    val ocrActive by ocr.activeId.collectAsState()
    val ocrStatus by ocr.status.collectAsState()

    LaunchedEffect(models) {
        if (downloadId != -1L) return@LaunchedEffect
        val savedId = preferences.getLong(Downloadable.PREF_MODEL_DL_ID, -1L)
        val savedPath = preferences.getString(Downloadable.PREF_MODEL_DL_PATH, "").orEmpty()
        if (savedId == -1L) return@LaunchedEffect
        val model = models.firstOrNull { it.destination?.path == savedPath }
        val cursor = dm.query(DownloadManager.Query().setFilterById(savedId))
        val status = if (cursor != null && cursor.moveToFirst()) {
            cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
        } else {
            -1
        }
        cursor?.close()
        when (status) {
            DownloadManager.STATUS_RUNNING,
            DownloadManager.STATUS_PENDING,
            DownloadManager.STATUS_PAUSED -> {
                downloadingModel = model
                downloadId = savedId
            }
            DownloadManager.STATUS_SUCCESSFUL -> {
                if (model != null && !model.isReady()) {
                    model.destination?.delete()
                    viewModel.log("上次下载未完成，已清理残缺文件，请重新下载")
                }
                clearDownloadPrefs(preferences)
            }
            else -> {
                if (model != null && !model.isReady()) {
                    model.destination?.delete()
                }
                clearDownloadPrefs(preferences)
            }
        }
    }

    LaunchedEffect(downloadId) {
        if (downloadId == -1L) return@LaunchedEffect
        while (true) {
            val cursor = dm.query(DownloadManager.Query().setFilterById(downloadId))
            if (cursor != null && cursor.moveToFirst()) {
                val status = cursor.getInt(
                    cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)
                )
                val bytesDownloaded = cursor.getLong(
                    cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                )
                val bytesTotal = cursor.getLong(
                    cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                )
                if (bytesTotal > 0) {
                    downloadProgress = bytesDownloaded.toDouble() / bytesTotal
                }
                when (status) {
                    DownloadManager.STATUS_SUCCESSFUL -> {
                        cursor.close()
                        val model = downloadingModel
                        downloadingModel = null
                        downloadId = -1L
                        downloadProgress = 0.0
                        clearDownloadPrefs(preferences)
                        delay(300)
                        if (model != null && model.isReady()) {
                            viewModel.clear()
                            viewModel.promptMode = promptModeForModel(model.name)
                            viewModel.load(model.destination!!.path, model.isHetero)
                        } else {
                            model?.destination?.delete()
                            viewModel.log("下载结束但文件不完整，请再点一次重新下载")
                        }
                        break
                    }
                    DownloadManager.STATUS_FAILED -> {
                        cursor.close()
                        downloadingModel?.let { model ->
                            if (!model.isReady()) {
                                model.destination?.delete()
                            }
                        }
                        viewModel.log("下载失败，请检查网络后重试")
                        downloadingModel = null
                        downloadId = -1L
                        downloadProgress = 0.0
                        clearDownloadPrefs(preferences)
                        break
                    }
                }
            } else {
                cursor?.close()
                downloadingModel = null
                downloadId = -1L
                downloadProgress = 0.0
                clearDownloadPrefs(preferences)
                break
            }
            cursor.close()
            delay(1000)
        }
    }

    fun startOrResumeDownload(selected: Downloadable, hetero: Boolean) {
        if (selected.isReady()) {
            viewModel.clear()
            viewModel.promptMode = promptModeForModel(selected.name)
            viewModel.load(selected.destination!!.path, hetero)
            return
        }
        val dest = selected.destination ?: return
        val source = selected.source ?: return
        if (downloadingModel != null) return
        dm.cancelDownloadsFor(dest, source)
        val staleId = preferences.getLong(Downloadable.PREF_MODEL_DL_ID, -1L)
        if (staleId != -1L) {
            dm.remove(staleId)
        }
        if (dest.exists()) {
            viewModel.log("发现未下完的文件（${dest.length()} 字节），已删除，重新下载")
            dest.delete()
        }
        dest.parentFile?.mkdirs()
        val request = DownloadManager.Request(source)
            .setTitle(selected.name)
            .setDescription("正在下载模型...")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationUri(dest.toUri())
            .setAllowedNetworkTypes(
                DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE
            )
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)
        downloadId = dm.enqueue(request)
        downloadingModel = selected
        preferences.edit()
            .putLong(Downloadable.PREF_MODEL_DL_ID, downloadId)
            .putString(Downloadable.PREF_MODEL_DL_PATH, dest.path)
            .putString(Downloadable.PREF_MODEL_DL_NAME, selected.name)
            .apply()
        viewModel.log(
            if (hetero) "开始下载草稿模型：${selected.name}" else "开始下载模型：${selected.name}"
        )
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
            Text("CPU 统计", style = MaterialTheme.typography.titleMedium)
            Text(
                text = cpuMeterHint(viewModel.cpuMeterMode, viewModel.cpuSourceLabel),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            CpuMeterMode.entries.forEach { mode ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            preferences.edit().putString("cpu_meter_mode", mode.stored()).apply()
                            viewModel.updateCpuMeterMode(mode)
                        },
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(
                        selected = viewModel.cpuMeterMode == mode,
                        onClick = {
                            preferences.edit().putString("cpu_meter_mode", mode.stored()).apply()
                            viewModel.updateCpuMeterMode(mode)
                        },
                    )
                    Text(cpuMeterTitle(mode))
                }
            }
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            Text("检验单识别", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            Text(
                text = ocrStatus,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(
                onClick = { scope.launch { ocr.setPreferred(OcrBackend.ONNX_V6) } },
                enabled = ocrActive != OcrBackend.ONNX_V6,
            ) {
                Text("使用 PP-OCRv6（ONNX）")
            }
            TextButton(
                onClick = { scope.launch { ocr.setPreferred(OcrBackend.LITE_V5) } },
                enabled = ocrActive != OcrBackend.LITE_V5,
            ) {
                Text("切回 PP-OCRv5（当前可回退版本）")
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
                                "Qwen3.8-27B-FP8" -> ApiType.QWEN38
                                else -> ApiType.QWEN
                            },
                            model.name,
                        )
                    },
                )
            }
            ModelGroupHeader("仅用手机")
            models.filter { !it.isApiModel && !it.isHetero }.forEach { model ->
                ModelRow(
                    model = model,
                    downloadingModel = downloadingModel,
                    downloadProgress = downloadProgress,
                    isCurrent = viewModel.isSelectedModel(model),
                    onLoadOrDownload = { startOrResumeDownload(it, hetero = false) },
                )
            }
            ModelGroupHeader("云端协同")
            models.filter { it.isHetero }.forEach { model ->
                ModelRow(
                    model = model,
                    downloadingModel = downloadingModel,
                    downloadProgress = downloadProgress,
                    isCurrent = viewModel.isSelectedModel(model),
                    onLoadOrDownload = { startOrResumeDownload(it, hetero = true) },
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
    isCurrent: Boolean,
    onLoadOrDownload: (Downloadable) -> Unit,
) {
    val incomplete = model.destination?.exists() == true && !model.isReady()
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
                } else if (isCurrent) {
                    Text("当前", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                } else if (incomplete) {
                    Text(
                        "未下完，点此重下",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        },
        onClick = { onLoadOrDownload(model) },
        enabled = downloadingModel == null || downloadingModel == model,
    )
}

private fun cpuMeterTitle(mode: CpuMeterMode): String = when (mode) {
    CpuMeterMode.FREQUENCY -> "频率"
    CpuMeterMode.AUTO -> "自动"
    CpuMeterMode.SYSTEM -> "整机"
    CpuMeterMode.PROCESS -> "本进程"
}

private fun cpuMeterHint(mode: CpuMeterMode, resolved: String): String {
    val choice = when (mode) {
        CpuMeterMode.FREQUENCY -> "默认。按各核当前频率相对最低、最高频率的比例估算，也就是 CPU Stats 的算法。"
        CpuMeterMode.AUTO -> "先读 /proc/stat。小米 14 能读到整机时间；读不到再用频率。"
        CpuMeterMode.SYSTEM -> "优先读 /proc/stat。读不到时用各核当前频率相对最低、最高频率的比例。"
        CpuMeterMode.PROCESS -> "本进程按 CPU 时间 / 墙钟 / 核心数计算，不读 /proc/stat。"
    }
    return if (resolved.isBlank()) choice else "$choice 当前用的是$resolved。"
}

private fun clearDownloadPrefs(preferences: SharedPreferences) {
    preferences.edit()
        .remove(Downloadable.PREF_MODEL_DL_ID)
        .remove(Downloadable.PREF_MODEL_DL_PATH)
        .remove(Downloadable.PREF_MODEL_DL_NAME)
        .apply()
}
