package com.example.llama

import android.app.DownloadManager
import android.net.Uri
import android.util.Log
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.core.database.getLongOrNull
import androidx.core.net.toUri
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

data class Downloadable(
    val name: String,
    val source: Uri?,
    val destination: File?,
    val isApiModel: Boolean = false,
    val isHetero: Boolean = false
) {
    fun isReady(): Boolean {
        val file = destination ?: return false
        return file.isCompleteGguf(advertisedMinBytes(name, file.name))
    }

    companion object {
        const val PREF_MODEL_DL_ID = "model_dl_id"
        const val PREF_MODEL_DL_PATH = "model_dl_path"
        const val PREF_MODEL_DL_NAME = "model_dl_name"

        @JvmStatic
        private val tag: String? = this::class.qualifiedName

        sealed interface State
        data object Ready: State
        data class Downloading(val id: Long): State
        data class Downloaded(val downloadable: Downloadable): State
        data class Error(val message: String): State

        @JvmStatic
        @Composable
        fun Button(viewModel: MainViewModel, dm: DownloadManager, item: Downloadable) {
            var status: State by remember {
                mutableStateOf(
                    if (item.isReady()) Downloaded(item)
                    else Ready
                )
            }
            var progress by remember { mutableDoubleStateOf(0.0) }

            val coroutineScope = rememberCoroutineScope()

            suspend fun waitForDownload(result: Downloading, item: Downloadable): State {
                while (true) {
                    val cursor = dm.query(DownloadManager.Query().setFilterById(result.id))

                    if (cursor == null) {
                        Log.e(tag, "dm.query() returned null")
                        return Error("dm.query() returned null")
                    }

                    if (!cursor.moveToFirst() || cursor.count < 1) {
                        cursor.close()
                        Log.i(tag, "cursor.moveToFirst() returned false or cursor.count < 1, download canceled?")
                        return Ready
                    }

                    val pix = cursor.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                    val tix = cursor.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                    val six = cursor.getColumnIndex(DownloadManager.COLUMN_STATUS)
                    val sofar = cursor.getLongOrNull(pix) ?: 0
                    val total = cursor.getLongOrNull(tix) ?: -1
                    val dmStatus = if (six >= 0) cursor.getInt(six) else DownloadManager.STATUS_RUNNING
                    cursor.close()

                    if (total > 0) {
                        progress = sofar.toDouble() / total
                    }

                    when (dmStatus) {
                        DownloadManager.STATUS_SUCCESSFUL -> {
                            delay(300)
                            return if (item.isReady()) Downloaded(item) else Error("文件不完整")
                        }
                        DownloadManager.STATUS_FAILED -> return Error("下载失败")
                    }

                    delay(1000L)
                }
            }

            fun onClick() {
                when (val s = status) {
                    is Downloaded -> {
                        if (!item.isReady()) {
                            item.destination?.delete()
                            status = Ready
                            viewModel.log("模型文件不完整，已删除，请重新下载")
                            return
                        }
                        viewModel.load(item.destination?.path ?: "", item.isHetero)
                    }

                    is Downloading -> {
                        coroutineScope.launch {
                            status = waitForDownload(s, item)
                            val result = status
                            if (result is Error) {
                                item.destination?.delete()
                                viewModel.log(result.message)
                            }
                        }
                    }

                    else -> {
                        item.destination?.let { dm.cancelDownloadsFor(it, item.source) }
                        item.destination?.delete()

                        val request = DownloadManager.Request(item.source ?: Uri.EMPTY)
                        request.setTitle("Downloading model")
                        request.setDescription("Downloading model: ${item.name}")
                        request.setAllowedNetworkTypes(
                            DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE
                        )
                        request.setAllowedOverMetered(true)
                        request.setAllowedOverRoaming(true)
                        request.setDestinationUri(item.destination?.toUri())

                        viewModel.log("Saving ${item.name} to ${item.destination?.path}")
                        Log.i(tag, "Saving ${item.name} to ${item.destination?.path}")

                        val id = dm.enqueue(request)
                        status = Downloading(id)
                        onClick()
                    }
                }
            }

            Button(onClick = { onClick() }, enabled = status !is Downloading) {
                when (status) {
                    is Downloading -> Text(text = "Downloading ${(progress * 100).toInt()}%")
                    is Downloaded -> Text("Load ${item.name}")
                    is Ready -> Text("Download ${item.name}")
                    is Error -> Text("Download ${item.name}")
                }
            }
        }
    }
}

fun DownloadManager.cancelDownloadsFor(dest: File, source: Uri? = null) {
    val needlePath = dest.absolutePath
    val needleUri = dest.toURI().toString()
    val sourceStr = source?.toString()
    val cursor = query(DownloadManager.Query()) ?: return
    try {
        val localIdx = cursor.getColumnIndex(DownloadManager.COLUMN_LOCAL_URI)
        val uriIdx = cursor.getColumnIndex(DownloadManager.COLUMN_URI)
        val idIdx = cursor.getColumnIndex(DownloadManager.COLUMN_ID)
        if (idIdx < 0) return
        val ids = ArrayList<Long>()
        while (cursor.moveToNext()) {
            val local = if (localIdx >= 0) cursor.getString(localIdx) else null
            val remote = if (uriIdx >= 0) cursor.getString(uriIdx) else null
            val matchLocal = local != null && (
                local == needleUri || local.contains(needlePath) || local.endsWith(dest.name)
            )
            val matchRemote = sourceStr != null && remote == sourceStr
            if (matchLocal || matchRemote) {
                ids.add(cursor.getLong(idIdx))
            }
        }
        ids.forEach { remove(it) }
    } finally {
        cursor.close()
    }
}

private const val GIB = 1L shl 30

fun advertisedMinBytes(vararg labels: String): Long {
    val haystack = labels.filter { it.isNotBlank() }.joinToString(" ")
    val match = Regex("""(\d+(?:\.\d+)?)\s*GiB""", RegexOption.IGNORE_CASE).find(haystack)
    if (match != null) {
        val gib = match.groupValues[1].toDouble()
        return (gib * GIB * 0.75).toLong().coerceAtLeast(32L shl 20)
    }
    val hintGiB = when {
        haystack.contains("Qwen3.5-0.8B", ignoreCase = true) &&
            haystack.contains("Q4", ignoreCase = true) -> 0.5
        haystack.contains("MiniCPM5-1B", ignoreCase = true) &&
            haystack.contains("Q4", ignoreCase = true) -> 0.6
        haystack.contains("Qwen3-0.6B", ignoreCase = true) &&
            (haystack.contains("BF16", ignoreCase = true) ||
                haystack.contains("FP16", ignoreCase = true) ||
                haystack.contains("f16", ignoreCase = true)) -> 1.2
        haystack.contains("Qwen3-0.6B", ignoreCase = true) -> 0.4
        haystack.contains("0.5B", ignoreCase = true) &&
            (haystack.contains("f16", ignoreCase = true) ||
                haystack.contains("FP16", ignoreCase = true)) -> 1.0
        haystack.contains("0.5B", ignoreCase = true) -> 0.4
        else -> 0.0
    }
    if (hintGiB > 0.0) return (hintGiB * GIB * 0.75).toLong()
    return 32L shl 20
}

fun File.isCompleteGguf(minBytes: Long): Boolean {
    if (!isFile) return false
    if (length() < minBytes) return false
    return try {
        inputStream().use { ins ->
            val magic = ByteArray(4)
            ins.read(magic) == 4 &&
                magic[0] == 'G'.code.toByte() &&
                magic[1] == 'G'.code.toByte() &&
                magic[2] == 'U'.code.toByte() &&
                magic[3] == 'F'.code.toByte()
        }
    } catch (_: Exception) {
        false
    }
}
