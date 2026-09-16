package com.example.llama.ocr

import android.content.Context
import android.graphics.Bitmap
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

internal class OnnxV6Engine(private val context: Context) : OcrEngine {
    override val id: String = OcrBackend.ONNX_V6
    private var paddle: PaddleOCR? = null

    override suspend fun init() = withContext(Dispatchers.IO) {
        val app = context.applicationContext
        if (!OpenCVUtils.init(app)) {
            throw IllegalStateException(
                OpenCVUtils.lastError ?: "OpenCV native library failed to load",
            )
        }
        paddle = PaddleOCR.create(
            app,
            PaddleOCRConfig(),
            EngineConfig(numThreads = 4),
            OcrBackend.V6_DET_ASSET,
            OcrBackend.V6_REC_ASSET,
            OcrBackend.V6_REC_YAML_ASSET,
        )
    }

    override suspend fun recognize(bitmap: Bitmap): String {
        val engine = paddle ?: throw IllegalStateException("PP-OCRv6 is not loaded")
        val result = engine.recognize(bitmap)
        return result.results.joinToString("\n") { it.text }
    }

    override fun release() {
        val engine = paddle
        paddle = null
        if (engine != null) {
            runBlocking { engine.release() }
        }
    }
}
