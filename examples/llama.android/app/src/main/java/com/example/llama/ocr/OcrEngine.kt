package com.example.llama.ocr

import android.graphics.Bitmap

internal interface OcrEngine {
    val id: String
    suspend fun init()
    suspend fun recognize(bitmap: Bitmap): String
    fun release()
}
