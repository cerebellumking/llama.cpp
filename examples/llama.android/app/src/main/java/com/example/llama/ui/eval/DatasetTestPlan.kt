package com.example.llama.ui.eval

import com.example.llama.InferenceMode

const val DATASET_STRATEGY_FIXED3 = "fixed3"
const val DATASET_STRATEGY_ADAPTIVE = "adaptive"
const val DATASET_STRATEGY_BOTH = "both"

/**
 * 只有协同解码才跑草稿 k。手机模型和在线模型各测一遍，不改 k。
 * 返回 null 表示这一遍不要动当前的草稿策略。
 */
fun datasetTestPasses(mode: InferenceMode, strategy: String): List<String?> {
    if (mode != InferenceMode.HETERO) return listOf(null)
    return when (strategy.trim().lowercase()) {
        DATASET_STRATEGY_BOTH -> listOf(DATASET_STRATEGY_FIXED3, DATASET_STRATEGY_ADAPTIVE)
        DATASET_STRATEGY_ADAPTIVE -> listOf(DATASET_STRATEGY_ADAPTIVE)
        else -> listOf(DATASET_STRATEGY_FIXED3)
    }
}

fun visibleDatasets(names: Array<String>?): List<String> =
    names.orEmpty().filter { it.isNotBlank() && !it.startsWith(".") }.sorted()
