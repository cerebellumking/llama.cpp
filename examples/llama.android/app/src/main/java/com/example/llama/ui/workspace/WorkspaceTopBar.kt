package com.example.llama.ui.workspace

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.llama.ConnectionState
import com.example.llama.InferenceMode
import com.example.llama.ui.theme.BluePrimary
import com.example.llama.ui.theme.BluePrimaryLight

@Composable
fun WorkspaceTopBar(
    connection: ConnectionState,
    inferenceMode: InferenceMode,
    evalUnlocked: Boolean,
    onTitleTap: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenTest: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(BluePrimary, BluePrimaryLight),
                ),
            ),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(
                    WindowInsets.safeDrawing.only(
                        WindowInsetsSides.Top + WindowInsetsSides.Horizontal,
                    ),
                )
                .height(56.dp)
                .padding(horizontal = 4.dp),
        ) {
            IconButton(
                onClick = onOpenTest,
                modifier = Modifier.align(Alignment.CenterStart),
            ) {
                Icon(
                    imageVector = Icons.Default.PlayArrow,
                    contentDescription = "测试",
                    tint = Color.White,
                )
            }

            Row(
                modifier = Modifier
                    .align(Alignment.Center)
                    .clickable(onClick = onTitleTap)
                    .padding(horizontal = 8.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "健康助手",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                )
                ConnectionChip(connection = connection, inferenceMode = inferenceMode)
            }

            Row(
                modifier = Modifier.align(Alignment.CenterEnd),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (evalUnlocked) {
                    Text(
                        text = "实验室",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White.copy(alpha = 0.9f),
                        modifier = Modifier.padding(end = 4.dp),
                    )
                }
                IconButton(onClick = onOpenSettings) {
                    Icon(
                        Icons.Default.Settings,
                        contentDescription = "设置",
                        tint = Color.White,
                    )
                }
            }
        }
    }
}

@Composable
private fun ConnectionChip(
    connection: ConnectionState,
    inferenceMode: InferenceMode,
) {
    val (label, color) = when {
        inferenceMode == InferenceMode.HETERO && connection == ConnectionState.Online ->
            "已连接" to Color.White
        inferenceMode == InferenceMode.HETERO && connection == ConnectionState.Offline ->
            "未连接" to Color(0xFFFFD6D6)
        inferenceMode == InferenceMode.API ->
            "在线" to Color.White
        else ->
            "手机" to Color.White.copy(alpha = 0.9f)
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(color),
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = color,
        )
    }
}
