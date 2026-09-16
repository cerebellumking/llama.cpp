package com.example.llama.ui.workspace

import android.app.DownloadManager
import android.content.Context
import android.llama.cpp.LLamaAndroid
import android.os.SystemClock
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import com.example.llama.Downloadable
import com.example.llama.MainViewModel
import com.example.llama.MessageKind
import com.example.llama.ocr.DualOcrController
import com.example.llama.ui.eval.DRAFT_STRATEGY_BOTH
import com.example.llama.ui.eval.EvalRail
import com.example.llama.ui.eval.EvalSettingsSheet
import com.example.llama.ui.eval.MetricsBar
import com.example.llama.ui.eval.TestDatasetDialog
import com.example.llama.ui.eval.runDatasetTest
import com.example.llama.ui.ocr.OcrReviewSheet
import kotlinx.coroutines.launch

@Composable
fun ReportWorkspaceRoute(
    viewModel: MainViewModel,
    dm: DownloadManager,
    models: List<Downloadable>,
    ocr: DualOcrController,
    onCamera: () -> Unit,
    onPickImage: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val preferences = remember {
        context.getSharedPreferences("heterospec_settings", Context.MODE_PRIVATE)
    }
    var evalUnlocked by rememberSaveable { mutableStateOf(false) }
    var titleTaps by remember { mutableIntStateOf(0) }
    var lastTapAt by remember { mutableLongStateOf(0L) }
    var showSettings by remember { mutableStateOf(false) }
    var showTestDialog by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        val savedUrl = preferences.getString("heterospec_server_url", null)
        if (!savedUrl.isNullOrBlank()) {
            viewModel.updateHeteroServerUrl(savedUrl)
        }
    }

    val onOpenSettings = remember { { showSettings = true } }
    val onOpenTest = remember { { showTestDialog = true } }
    val onTitleTap = remember {
        {
            val now = SystemClock.elapsedRealtime()
            titleTaps = if (now - lastTapAt < 700L) titleTaps + 1 else 1
            lastTapAt = now
            if (titleTaps >= 5) {
                evalUnlocked = !evalUnlocked
                titleTaps = 0
            }
        }
    }
    val events = remember(viewModel, onCamera, onPickImage, onOpenSettings, onOpenTest, onTitleTap) {
        WorkspaceEvents(
            onInputChange = viewModel::updateMessage,
            onSend = { viewModel.send() },
            onStop = viewModel::stopGeneration,
            onCamera = onCamera,
            onPickImage = onPickImage,
            onEditMessage = viewModel::startEditingMessage,
            onUpdateEditingText = viewModel::updateEditingText,
            onConfirmEdit = viewModel::confirmEditMessage,
            onCancelEdit = viewModel::cancelEditingMessage,
            onOcrTextChange = viewModel::updateOcrText,
            onConfirmOcr = viewModel::confirmOcrAndSend,
            onCancelOcr = viewModel::hideOcrEditor,
            onRetry = viewModel::retryLast,
            onDismissError = viewModel::dismissError,
            onTitleTap = onTitleTap,
            onOpenSettings = onOpenSettings,
            onOpenTest = onOpenTest,
            onToggleCpu = viewModel::updateCpuMonitorEnabled,
        )
    }

    Column(modifier = modifier.fillMaxSize()) {
        LiveTopBar(
            viewModel = viewModel,
            evalUnlocked = evalUnlocked,
            onTitleTap = onTitleTap,
            onOpenSettings = onOpenSettings,
            onOpenTest = onOpenTest,
        )
        LiveMetricsBar(viewModel)
        if (evalUnlocked) {
            LiveEvalRail(viewModel, onOpenTest)
        }
        LiveFeed(
            viewModel = viewModel,
            events = events,
            modifier = Modifier.weight(1f),
        )
        LiveComposerBar(viewModel, events)
    }
    LiveOcrSheet(viewModel, events)

    if (showSettings) {
        EvalSettingsSheet(
            viewModel = viewModel,
            models = models,
            dm = dm,
            evalUnlocked = evalUnlocked,
            ocr = ocr,
            onDismiss = { showSettings = false },
        )
    }

    if (showTestDialog) {
        TestDatasetDialog(
            onDismiss = { showTestDialog = false },
            onSelectDataset = { datasetName, strategy ->
                showTestDialog = false
                scope.launch {
                    val strategies = if (strategy == DRAFT_STRATEGY_BOTH) {
                        listOf(
                            LLamaAndroid.DRAFT_STRATEGY_FIXED3,
                            LLamaAndroid.DRAFT_STRATEGY_ADAPTIVE,
                        )
                    } else {
                        listOf(LLamaAndroid.normalizeDraftStrategy(strategy))
                    }
                    for (s in strategies) {
                        viewModel.updateHeteroDraftStrategy(s)
                        runDatasetTest(
                            context,
                            viewModel,
                            datasetName,
                            ocr,
                            strategyLabel = s,
                        )
                    }
                }
            },
        )
    }
}

@Composable
private fun LiveTopBar(
    viewModel: MainViewModel,
    evalUnlocked: Boolean,
    onTitleTap: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenTest: () -> Unit,
) {
    WorkspaceTopBar(
        connection = viewModel.connectionState,
        inferenceMode = viewModel.inferenceMode,
        evalUnlocked = evalUnlocked,
        onTitleTap = onTitleTap,
        onOpenSettings = onOpenSettings,
        onOpenTest = onOpenTest,
    )
}

@Composable
private fun LiveMetricsBar(viewModel: MainViewModel) {
    MetricsBar(
        metrics = EvalMetrics(
            inferenceSpeed = viewModel.inferenceSpeed,
            isInferenceRunning = viewModel.isInferenceRunning,
            cpuUsage = viewModel.cpuUsage,
            cpuEnabled = viewModel.isCpuMonitorEnabled,
            datasetSpeed = viewModel.datasetSpeed,
            lastTestSpeed = viewModel.lastTestSpeed,
            testRunning = viewModel.isTestRunning,
            testProcessed = viewModel.testProcessed,
            testTotal = viewModel.testTotal,
        ),
        onToggleCpu = viewModel::updateCpuMonitorEnabled,
    )
}

@Composable
private fun LiveEvalRail(viewModel: MainViewModel, onOpenTest: () -> Unit) {
    val logs by remember {
        derivedStateOf { viewModel.messages.filter { it.kind == MessageKind.EvalLog } }
    }
    EvalRail(
        metrics = EvalMetrics(
            draftStrategy = viewModel.heteroDraftStrategy,
            gpuQueueDepth = viewModel.gpuQueueDepth,
            logs = logs,
        ),
        onOpenTest = onOpenTest,
    )
}

@Composable
private fun LiveFeed(
    viewModel: MainViewModel,
    events: WorkspaceEvents,
    modifier: Modifier = Modifier,
) {
    val feed by remember {
        derivedStateOf { viewModel.messages.filter { it.kind != MessageKind.EvalLog } }
    }
    ReportWorkspaceContent(
        ui = WorkspaceUi(
            feed = feed,
            isInferenceRunning = viewModel.isInferenceRunning,
            canSend = viewModel.canSend,
            editingMessageId = viewModel.editingMessageId,
            editingText = viewModel.editingText,
            connection = viewModel.connectionState,
            inferenceMode = viewModel.inferenceMode,
            errorBanner = viewModel.errorBanner,
        ),
        events = events,
        modifier = modifier,
    )
}

@Composable
private fun LiveComposerBar(viewModel: MainViewModel, events: WorkspaceEvents) {
    val hasReport by remember {
        derivedStateOf { viewModel.messages.any { it.kind == MessageKind.Report } }
    }
    ComposerBar(
        input = viewModel.message,
        onInputChange = events.onInputChange,
        isInferenceRunning = viewModel.isInferenceRunning,
        sendEnabled = viewModel.canSend && (viewModel.message.isNotBlank() || hasReport),
        placeholder = if (viewModel.canSend) "补充说明（可选）" else "请先到设置里选择模型",
        onSend = events.onSend,
        onStop = events.onStop,
        onCamera = events.onCamera,
        onPickImage = events.onPickImage,
    )
}

@Composable
private fun LiveOcrSheet(viewModel: MainViewModel, events: WorkspaceEvents) {
    if (!viewModel.isShowingOcrEditor) return
    OcrReviewSheet(
        text = viewModel.ocrEditText,
        image = viewModel.ocrSourceImage,
        onTextChange = events.onOcrTextChange,
        onConfirm = events.onConfirmOcr,
        onCancel = events.onCancelOcr,
    )
}
