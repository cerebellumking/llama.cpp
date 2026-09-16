package com.example.llama.ocr

import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class DualOcrController(
    context: Context,
    private val prefs: SharedPreferences,
    private val log: (String) -> Unit,
) {
    private val appContext = context.applicationContext
    private val mutex = Mutex()
    private val lite = LiteV5Engine(appContext)
    private var v6: OnnxV6Engine? = null
    private val started = CompletableDeferred<Unit>()

    private val _activeId = MutableStateFlow(OcrBackend.LITE_V5)
    val activeId: StateFlow<String> = _activeId

    private val _status = MutableStateFlow("正在加载 OCR…")
    val status: StateFlow<String> = _status

    suspend fun start() {
        try {
            mutex.withLock {
                lite.init()
                log("OCR 回退引擎已就绪：${OcrBackend.displayName(OcrBackend.LITE_V5)}")
                val preferred = prefs.getString(OcrBackend.KEY_PREFERRED, OcrBackend.DEFAULT)
                    ?: OcrBackend.DEFAULT
                if (preferred == OcrBackend.ONNX_V6) {
                    activateV6Locked()
                } else {
                    _activeId.value = OcrBackend.LITE_V5
                    _status.value = "当前 ${OcrBackend.displayName(OcrBackend.LITE_V5)}"
                    log("OCR 当前：${OcrBackend.displayName(OcrBackend.LITE_V5)}")
                }
            }
            started.complete(Unit)
        } catch (t: Throwable) {
            _status.value = "OCR 初始化失败：${t.message}"
            started.completeExceptionally(t)
            throw t
        }
    }

    suspend fun setPreferred(id: String) {
        started.await()
        mutex.withLock {
            prefs.edit().putString(OcrBackend.KEY_PREFERRED, id).apply()
            if (id == OcrBackend.ONNX_V6) {
                activateV6Locked()
            } else {
                _activeId.value = OcrBackend.LITE_V5
                _status.value = "当前 ${OcrBackend.displayName(OcrBackend.LITE_V5)}"
                log("OCR 已切换回 ${OcrBackend.displayName(OcrBackend.LITE_V5)}")
            }
        }
    }

    suspend fun recognize(bitmap: Bitmap): String {
        started.await()
        return mutex.withLock {
            if (_activeId.value == OcrBackend.ONNX_V6) {
                val engine = v6
                if (engine != null) {
                    try {
                        return@withLock engine.recognize(bitmap)
                    } catch (t: Throwable) {
                        _activeId.value = OcrBackend.LITE_V5
                        _status.value = "v6 识别失败，已改用 v5"
                        log("PP-OCRv6 识别失败，改用 Lite v5：${t.message}")
                    }
                }
            }
            lite.recognize(bitmap)
        }
    }

    fun release() {
        v6?.release()
        v6 = null
        lite.release()
    }

    private suspend fun activateV6Locked() {
        try {
            if (v6 == null) {
                val engine = OnnxV6Engine(appContext)
                engine.init()
                v6 = engine
            }
            _activeId.value = OcrBackend.ONNX_V6
            _status.value = "当前 ${OcrBackend.displayName(OcrBackend.ONNX_V6)}（可切回 v5）"
            log("OCR 当前：${OcrBackend.displayName(OcrBackend.ONNX_V6)}")
        } catch (t: Throwable) {
            _activeId.value = OcrBackend.LITE_V5
            _status.value = "v6 加载失败，已回退 v5：${t.message}"
            log("PP-OCRv6 加载失败，已回退 Lite v5：${t.message}")
            t.cause?.let { log("PP-OCRv6 原因：${it.message}") }
        }
    }
}
