package com.example.llama.ui.eval

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.example.llama.MainViewModel
import com.example.llama.cleanOcrText
import com.example.llama.ocr.DualOcrController
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * @return false when the user stopped the run. The caller must not start the next pass.
 */
suspend fun runDatasetTest(
    context: Context,
    viewModel: MainViewModel,
    datasetName: String,
    ocr: DualOcrController,
    strategyLabel: String? = null,
): Boolean {
    val base = "datasets/$datasetName"
    val entries = try {
        context.assets.list(base)
    } catch (_: Exception) {
        null
    }

    val imageExts = setOf(".jpg", ".jpeg", ".png", ".webp", ".bmp")

    if (entries != null && entries.isNotEmpty()) {
        val imageFiles = entries.filter { name ->
            val lower = name.lowercase(Locale.ROOT)
            imageExts.any { lower.endsWith(it) }
        }.sorted()

        if (imageFiles.isEmpty()) {
            viewModel.log("数据集里没有图片：$datasetName")
            return true
        }
        viewModel.startTest(imageFiles.size, strategyLabel)
        for (file in imageFiles) {
            coroutineContext.ensureActive()
            if (!viewModel.isTestRunning) return false
            try {
                val bitmap = withContext(Dispatchers.Default) {
                    context.assets.open("$base/$file").use { input ->
                        BitmapFactory.decodeStream(input)
                    }
                }
                if (bitmap != null) {
                    val text = try {
                        runOcrSuspend(ocr, bitmap)
                    } finally {
                        if (!bitmap.isRecycled) bitmap.recycle()
                    }
                    viewModel.sendForTest_impl(text)
                } else {
                    viewModel.skipTestItem()
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                viewModel.log("读取图片失败：${e.message}")
                viewModel.skipTestItem()
            }
        }
        if (!viewModel.isTestRunning) return false
        viewModel.finishTest(strategyLabel)
        return true
    }

    val candidates = listOf("$base.json", "$base.csv", "$base.txt")
    var openedName: String? = null
    for (c in candidates) {
        try {
            context.assets.open(c).close()
            openedName = c
            break
        } catch (_: Exception) {
        }
    }

    if (openedName == null) {
        viewModel.log("未识别的数据集：$datasetName")
        return true
    }

    val prompts: List<String> = when {
        openedName.endsWith(".json") -> readPromptsFromJson(context, openedName)
        openedName.endsWith(".csv") -> readPromptsFromCsv(context, openedName)
        else -> readPromptsFromTxt(context, openedName)
    }

    if (prompts.isEmpty()) {
        viewModel.log("数据集是空的：$datasetName")
        return true
    }
    viewModel.startTest(prompts.size, strategyLabel)
    for (p in prompts) {
        coroutineContext.ensureActive()
        if (!viewModel.isTestRunning) return false
        viewModel.sendForTest_impl(p)
    }
    if (!viewModel.isTestRunning) return false
    viewModel.finishTest(strategyLabel)
    return true
}

private fun readPromptsFromTxt(context: Context, assetPath: String): List<String> =
    try {
        context.assets.open(assetPath).use { input ->
            BufferedReader(InputStreamReader(input)).readLines().map { it.trim() }.filter { it.isNotEmpty() }
        }
    } catch (_: Exception) {
        emptyList()
    }

private fun readPromptsFromCsv(context: Context, assetPath: String): List<String> =
    try {
        context.assets.open(assetPath).use { input ->
            val lines = BufferedReader(InputStreamReader(input)).readLines()
            if (lines.isEmpty()) return emptyList()
            val header = lines.first().split(',')
            val colIndex = header.indexOfFirst {
                it.equals("question", true) || it.equals("prompt", true) || it.equals("input", true)
            }
            val rows = lines.drop(1)
            rows.mapNotNull { row ->
                val cols = row.split(',')
                val value = if (colIndex in cols.indices) cols[colIndex] else cols.firstOrNull()
                value?.trim()?.takeIf { it.isNotEmpty() }
            }
        }
    } catch (_: Exception) {
        emptyList()
    }

private fun readPromptsFromJson(context: Context, assetPath: String): List<String> =
    try {
        val text = context.assets.open(assetPath).use { input ->
            BufferedReader(InputStreamReader(input)).readText()
        }
        val arr = JSONArray(text)
        val result = mutableListOf<String>()
        for (i in 0 until arr.length()) {
            val el = arr.get(i)
            when (el) {
                is JSONObject -> {
                    val v = el.optString(
                        "question",
                        el.optString("prompt", el.optString("input", "")),
                    )
                    if (v.isNotBlank()) result += v
                }
                is String -> if (el.isNotBlank()) result += el
            }
        }
        result
    } catch (_: Exception) {
        emptyList()
    }

private suspend fun runOcrSuspend(ocr: DualOcrController, bitmap: Bitmap): String =
    try {
        cleanOcrText(ocr.recognize(bitmap))
    } catch (e: CancellationException) {
        throw e
    } catch (_: Exception) {
        ""
    }
