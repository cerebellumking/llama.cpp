package com.example.llama.ocr

object OcrBackend {
    const val PREFS = "heterospec_settings"
    const val KEY_PREFERRED = "ocr_engine"
    const val LITE_V5 = "lite_v5"
    const val ONNX_V6 = "onnx_v6"
    const val DEFAULT = ONNX_V6

    const val V6_DET_ASSET = "models/ppocrv6/det/inference.onnx"
    const val V6_REC_ASSET = "models/ppocrv6/rec/inference.onnx"
    const val V6_REC_YAML_ASSET = "models/ppocrv6/rec/inference.yml"

    fun displayName(id: String): String = when (id) {
        ONNX_V6 -> "ONNX / PP-OCRv6_small"
        else -> "Lite / PP-OCRv5"
    }
}
