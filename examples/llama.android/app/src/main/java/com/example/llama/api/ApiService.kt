package com.example.llama.api

import kotlinx.coroutines.flow.Flow

interface ApiService {
    fun send(prompt: String): Flow<Pair<String, Int>>
    companion object {
        fun getInstance(type: ApiType): ApiService {
            return when (type) {
                ApiType.DEEPSEEK -> DeepseekApiService.getInstance()
                ApiType.QWEN -> QwenApiService.getInstance()
                ApiType.HETEROSPEC -> HeteroSpecService.getInstance()
            }
        }
    }
}

enum class ApiType {
    DEEPSEEK,
    QWEN,
    HETEROSPEC
} 