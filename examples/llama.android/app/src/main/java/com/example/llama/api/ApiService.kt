package com.example.llama.api

import kotlinx.coroutines.flow.Flow

interface ApiService {
    fun send(prompt: String): Flow<Pair<String, Int>>
    companion object {
        fun getInstance(type: ApiType): ApiService {
            return when (type) {
                ApiType.DEEPSEEK -> DeepseekApiService.getInstance()
                ApiType.QWEN, ApiType.QWEN38 -> QwenApiService.getInstance()
            }
        }
    }
}

// HeteroSpec 不走 ApiService，只通过 LLamaAndroid.sendHetero。
// HeteroSpec 不走 ApiService，只通过 LLamaAndroid.sendHetero。
enum class ApiType {
    DEEPSEEK,
    QWEN,
    QWEN38,
}
