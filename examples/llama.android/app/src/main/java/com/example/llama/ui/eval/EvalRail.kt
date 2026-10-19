package com.example.llama.ui.eval

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.llama.ui.theme.BluePrimary
import com.example.llama.ui.theme.BluePrimaryLight
import com.example.llama.ui.workspace.EvalMetrics
import java.util.Locale

private fun speedText(speed: Double): String {
    return String.format(Locale.US, "%.2f", speed) + " tokens/s"
}

private fun generationSpeedText(tpotMs: Double): String {
    return if (tpotMs <= 0.0) {
        "生成速度 --"
    } else {
        String.format(Locale.US, "生成速度 %.2f tokens/s", 1000.0 / tpotMs)
    }
}

@Composable
fun MetricsBar(
    metrics: EvalMetrics,
    onToggleCpu: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        BluePrimary.copy(alpha = 0.12f),
                        BluePrimaryLight.copy(alpha = 0.08f),
                    ),
                ),
            )
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(
                modifier = Modifier.weight(1f),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (metrics.testRunning || metrics.lastTestSpeed > 0) {
                    Icon(
                        imageVector = Icons.Default.Speed,
                        contentDescription = "测试速度",
                        tint = BluePrimary,
                        modifier = Modifier.size(16.dp),
                    )
                    Spacer(Modifier.width(6.dp))
                    val speed = if (metrics.testRunning) metrics.datasetSpeed else metrics.lastTestSpeed
                    val testLabel = "测试 " + speedText(speed) +
                        "  " + metrics.testProcessed + "/" + metrics.testTotal
                    Text(
                        text = testLabel,
                        color = BluePrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.Speed,
                        contentDescription = "生成速度",
                        tint = BluePrimary,
                        modifier = Modifier.size(16.dp),
                    )
                    Spacer(Modifier.width(6.dp))
                    Text(
                        text = generationSpeedText(metrics.requestTpotMs),
                        color = BluePrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                    )
                }
            }

            Icon(
                imageVector = Icons.Default.Memory,
                contentDescription = "CPU",
                tint = if (metrics.cpuEnabled) BluePrimary else BluePrimary.copy(alpha = 0.6f),
                modifier = Modifier.size(16.dp),
            )
            Spacer(Modifier.width(4.dp))
            Text(
                text = if (metrics.cpuEnabled) {
                    String.format(Locale.US, "%.0f%%", metrics.cpuUsage)
                } else {
                    "CPU 关"
                },
                color = if (metrics.cpuEnabled) BluePrimary else BluePrimary.copy(alpha = 0.8f),
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
            )
            Switch(
                checked = metrics.cpuEnabled,
                onCheckedChange = onToggleCpu,
            )
        }
    }
}

@Composable
fun EvalRail(
    metrics: EvalMetrics,
    onOpenTest: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val scheme = MaterialTheme.colorScheme
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(scheme.primaryContainer.copy(alpha = 0.55f))
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            IconButton(onClick = onOpenTest, modifier = Modifier.size(32.dp)) {
                Icon(Icons.Default.PlayArrow, contentDescription = "测试模式", tint = scheme.primary)
            }
            Text("数据集测试", style = MaterialTheme.typography.labelLarge, color = scheme.onPrimaryContainer)
            Spacer(Modifier.weight(1f))
            Text("k=${metrics.draftStrategy}", style = MaterialTheme.typography.labelSmall, color = scheme.onPrimaryContainer)
            metrics.gpuQueueDepth?.let { depth ->
                Spacer(Modifier.width(8.dp))
                Text("q=$depth", style = MaterialTheme.typography.labelSmall, color = scheme.onPrimaryContainer)
            }
        }
        if (metrics.logs.isNotEmpty()) {
            Spacer(Modifier.height(4.dp))
            LazyColumn(modifier = Modifier.heightIn(max = 88.dp)) {
                items(metrics.logs.takeLast(8), key = { it.id }) { log ->
                    Text(
                        text = log.content,
                        style = MaterialTheme.typography.labelSmall,
                        color = scheme.onPrimaryContainer,
                        maxLines = 1,
                    )
                }
            }
        }
    }
}
