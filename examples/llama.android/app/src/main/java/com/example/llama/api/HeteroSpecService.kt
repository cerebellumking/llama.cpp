package com.example.llama.api

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class HeteroSpecService : ApiService {
    companion object {
        private var instance: HeteroSpecService? = null
        
        fun getInstance(): HeteroSpecService {
            if (instance == null) {
                instance = HeteroSpecService()
            }
            return instance!!
        }
    }

    override fun send(prompt: String): Flow<Pair<String, Int>> = flow {
        // TODO: 实现HeteroSpec API的调用逻辑
        emit(Pair("HeteroSpec API response", 1))
    }
}
