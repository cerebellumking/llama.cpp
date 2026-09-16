package com.example.llama.ui.workspace

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

data class ThinkParse(
    val think: String?,
    val reply: String,
)

fun parseThink(content: String): ThinkParse {
    val lower = content.lowercase()
    val end = lower.indexOf("</think>")
    if (end == -1) return ThinkParse(null, content)
    val start = lower.indexOf("<think>")
    val think = if (start != -1) {
        content.substring(start + 7, end).trim()
    } else {
        content.substring(0, end).trim()
    }
    val reply = content.substring(end + 8).trim()
    return ThinkParse(think.ifEmpty { null }, reply)
}

@Composable
fun ReadingPane(
    content: String,
    modifier: Modifier = Modifier,
    streaming: Boolean = false,
) {
    val parsed = remember(content) { parseThink(content) }
    var showThinking by remember { mutableStateOf(false) }
    val scheme = MaterialTheme.colorScheme

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = scheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "解读",
                style = MaterialTheme.typography.labelSmall,
                color = scheme.primary,
            )
            Spacer(Modifier.size(8.dp))
            val think = parsed.think
            if (!think.isNullOrEmpty()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { showThinking = !showThinking }
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        imageVector = if (showThinking) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        contentDescription = if (showThinking) "收起思考" else "展开思考",
                        tint = scheme.onSurfaceVariant,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.width(6.dp))
                    Text(
                        text = "思考过程",
                        style = MaterialTheme.typography.labelLarge,
                        color = scheme.onSurfaceVariant,
                    )
                }
                AnimatedVisibility(visible = showThinking) {
                    ReportBody(
                        text = think,
                        streaming = streaming,
                        style = MaterialTheme.typography.bodyMedium,
                        color = scheme.onSurfaceVariant,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                    )
                }
            }
            val body = parsed.reply.ifBlank { content }
            if (body.isNotBlank()) {
                ReportBody(
                    text = body,
                    streaming = streaming,
                    style = MaterialTheme.typography.bodyLarge,
                    color = scheme.onSurface,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
private fun ReportBody(
    text: String,
    streaming: Boolean,
    style: TextStyle,
    color: Color,
    modifier: Modifier = Modifier,
) {
    if (streaming) {
        Text(text = text, modifier = modifier, style = style, color = color)
        return
    }
    val annotated = remember(text) { markdownToAnnotated(text) }
    Text(text = annotated, modifier = modifier, style = style, color = color)
}

internal fun markdownToAnnotated(markdown: String): AnnotatedString = buildAnnotatedString {
    val lines = markdown.split('\n')
    lines.forEachIndexed { index, raw ->
        if (index > 0) append('\n')
        val trimmed = raw.trimStart()
        when {
            trimmed.startsWith("### ") -> {
                withStyle(SpanStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp)) {
                    appendInlines(trimmed.substring(4))
                }
            }
            trimmed.startsWith("## ") -> {
                withStyle(SpanStyle(fontWeight = FontWeight.SemiBold, fontSize = 17.sp)) {
                    appendInlines(trimmed.substring(3))
                }
            }
            trimmed.startsWith("# ") -> {
                withStyle(SpanStyle(fontWeight = FontWeight.SemiBold, fontSize = 18.sp)) {
                    appendInlines(trimmed.substring(2))
                }
            }
            trimmed.startsWith("- ") || trimmed.startsWith("* ") -> {
                append("• ")
                appendInlines(trimmed.substring(2))
            }
            else -> appendInlines(raw)
        }
    }
}

private fun AnnotatedString.Builder.appendInlines(text: String) {
    var i = 0
    while (i < text.length) {
        when {
            text.startsWith("**", i) -> {
                val end = text.indexOf("**", i + 2)
                if (end == -1) {
                    append(text.substring(i))
                    return
                }
                withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) {
                    append(text.substring(i + 2, end))
                }
                i = end + 2
            }
            text[i] == '`' -> {
                val end = text.indexOf('`', i + 1)
                if (end == -1) {
                    append(text.substring(i))
                    return
                }
                withStyle(SpanStyle(fontFamily = FontFamily.Monospace, fontSize = 13.sp)) {
                    append(text.substring(i + 1, end))
                }
                i = end + 1
            }
            else -> {
                val nextBold = text.indexOf("**", i).let { if (it < 0) text.length else it }
                val nextCode = text.indexOf('`', i).let { if (it < 0) text.length else it }
                val next = minOf(nextBold, nextCode)
                append(text.substring(i, next))
                i = next
            }
        }
    }
}
