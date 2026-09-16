package com.example.llama.ui.workspace

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.tappableElement
import androidx.compose.foundation.layout.union
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp

@Composable
fun ComposerBar(
    input: String,
    onInputChange: (String) -> Unit,
    isInferenceRunning: Boolean,
    sendEnabled: Boolean,
    placeholder: String,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onCamera: () -> Unit,
    onPickImage: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val scheme = MaterialTheme.colorScheme
    Column(
        modifier = modifier
            .background(scheme.surface)
            .windowInsetsPadding(
                WindowInsets.safeDrawing
                    .union(WindowInsets.tappableElement)
                    .union(WindowInsets.ime)
                    .only(WindowInsetsSides.Bottom + WindowInsetsSides.Horizontal)
            ),
    ) {
        HorizontalDivider(thickness = 1.dp, color = scheme.outlineVariant)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.Bottom,
        ) {
            IconButton(
                onClick = onCamera,
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(scheme.surfaceVariant),
            ) {
                Icon(Icons.Default.Camera, contentDescription = "拍照", tint = scheme.primary)
            }
            Spacer(Modifier.width(8.dp))
            IconButton(
                onClick = onPickImage,
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(scheme.surfaceVariant),
            ) {
                Icon(Icons.Default.Image, contentDescription = "相册", tint = scheme.primary)
            }
            Spacer(Modifier.width(8.dp))
            OutlinedTextField(
                value = input,
                onValueChange = onInputChange,
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 48.dp),
                placeholder = {
                    Text(placeholder, color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
                },
                shape = RoundedCornerShape(14.dp),
                maxLines = 4,
            )
            Spacer(Modifier.width(8.dp))
            Button(
                onClick = if (isInferenceRunning) onStop else onSend,
                enabled = isInferenceRunning || sendEnabled,
                modifier = Modifier.height(48.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isInferenceRunning) scheme.error else scheme.primary,
                ),
                contentPadding = PaddingValues(horizontal = 16.dp),
            ) {
                Text(if (isInferenceRunning) "停止" else "发送")
            }
        }
    }
}
