package com.example.llama.ocr

import android.content.Context
import android.graphics.Bitmap
import com.equationl.paddleocr4android.CpuPowerMode
import com.equationl.paddleocr4android.OCR
import com.equationl.paddleocr4android.OcrConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal class LiteV5Engine(context: Context) : OcrEngine {
    override val id: String = OcrBackend.LITE_V5
    private val ocr = OCR(context.applicationContext)

    override suspend fun init() = withContext(Dispatchers.IO) {
        val config = OcrConfig().apply {
            modelPath = "models/ch_PP-OCRv5"
            labelPath = "labels/ppocr_keys_ocrv5.txt"
            isRunDet = true
            isRunCls = true
            isRunRec = true
            clsModelFilename = "cls.nb"
            detModelFilename = "det.nb"
            recModelFilename = "rec.nb"
            cpuPowerMode = CpuPowerMode.LITE_POWER_FULL
            isDrwwTextPositionBox = false
        }
        val ok = ocr.initModelSync(config).getOrThrow()
        if (!ok) throw IllegalStateException("Lite PP-OCRv5 init returned false")
    }

    override suspend fun recognize(bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        ocr.runSync(bitmap).getOrThrow().simpleText
    }

    override fun release() {
        ocr.releaseModel()
    }
}
