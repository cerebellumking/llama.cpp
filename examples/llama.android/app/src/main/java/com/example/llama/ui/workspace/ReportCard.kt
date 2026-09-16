package com.example.llama.ui.workspace

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog

@Composable
fun ReportCard(
    content: String,
    image: Bitmap?,
    isEditing: Boolean,
    editingText: String,
    onEdit: () -> Unit,
    onUpdateEditingText: (String) -> Unit,
    onConfirmEdit: () -> Unit,
    onCancelEdit: () -> Unit,
    modifier: Modifier = Modifier,
    actions: @Composable RowScope.() -> Unit = {},
) {
    var showViewer by remember { mutableStateOf(false) }
    val clipboard = LocalClipboardManager.current
    val scheme = MaterialTheme.colorScheme

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = scheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, scheme.outlineVariant),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = "检验单",
                style = MaterialTheme.typography.labelSmall,
                color = scheme.primary,
            )
            if (image != null) {
                val preview = remember(image) { image.asImageBitmap() }
                Spacer(Modifier.size(8.dp))
                Image(
                    bitmap = preview,
                    contentDescription = "报告图片",
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 180.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .clickable { showViewer = true },
                    contentScale = ContentScale.Fit,
                )
            }
            Spacer(Modifier.size(8.dp))
            if (isEditing) {
                OutlinedTextField(
                    value = editingText,
                    onValueChange = onUpdateEditingText,
                    modifier = Modifier.fillMaxWidth(),
                    maxLines = 8,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                ) {
                    TextButton(onClick = onCancelEdit) { Text("取消") }
                    TextButton(onClick = onConfirmEdit) { Text("发送") }
                }
            } else {
                if (content.isNotBlank()) {
                    Text(
                        text = content,
                        style = MaterialTheme.typography.bodyLarge,
                        color = scheme.onSurface,
                        maxLines = 8,
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (content.isNotBlank()) {
                        IconButton(onClick = { clipboard.setText(AnnotatedString(content)) }) {
                            Icon(
                                Icons.Default.ContentCopy,
                                contentDescription = "复制",
                                modifier = Modifier.size(18.dp),
                                tint = scheme.onSurfaceVariant,
                            )
                        }
                        IconButton(onClick = onEdit) {
                            Icon(
                                Icons.Default.Edit,
                                contentDescription = "编辑",
                                modifier = Modifier.size(18.dp),
                                tint = scheme.onSurfaceVariant,
                            )
                        }
                    }
                    actions()
                }
            }
        }
    }

    if (showViewer && image != null) {
        ImageViewerDialog(bitmap = image, onDismiss = { showViewer = false })
    }
}

@Composable
fun ImageViewerDialog(
    bitmap: Bitmap,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Dialog(onDismissRequest = onDismiss) {
        Box(
            modifier = modifier
                .fillMaxWidth()
                .fillMaxHeight(0.8f)
                .background(Color.Black)
                .padding(16.dp),
        ) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Full size image",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
            IconButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp),
            ) {
                Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.White)
            }
        }
    }
}
