package com.example.llama.api

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class DeepseekApiService {
    private val tag = "DeepseekApiService"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private var apiKey: String = ""

    companion object {
        private val instance = DeepseekApiService()
        fun getInstance(): DeepseekApiService = instance

        private const val API_URL = "https://api.deepseek.com/chat/completions"
        private const val MODEL_NAME = "deepseek-chat"
    }

    fun setApiKey(key: String) {
        apiKey = key
    }

    // 使用挂起函数处理OkHttp请求
    private suspend fun executeRequest(request: Request): Response {
        return suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)

            continuation.invokeOnCancellation {
                call.cancel()
            }

            call.enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    continuation.resumeWithException(e)
                }

                override fun onResponse(call: Call, response: Response) {
                    continuation.resume(response)
                }
            })
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    fun send(userMessage: String, systemPrompt: String = "You are a helpful assistant."): Flow<Pair<String, Int>> = callbackFlow {
        if (apiKey.isEmpty()) {
            throw IOException("API Key not set. Please set your DeepSeek API key.")
        }

        val jsonBody = JSONObject().apply {
            put("model", MODEL_NAME)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", systemPrompt)
                })
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", userMessage)
                })
            })
            put("stream", true)
        }

        val requestBody = jsonBody.toString().toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(API_URL)
            .addHeader("Content-Type", "application/json")
            .addHeader("Authorization", "Bearer $apiKey")
            .post(requestBody)
            .build()

        var isFirstToken = true
        val call = client.newCall(request)

        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(tag, "API request failed: ${e.message}", e)
                close(e)
            }

            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    val errorBody = response.body?.string() ?: "Unknown error"
                    val error = IOException("API error: ${response.code} - $errorBody")
                    Log.e(tag, "API error: ${response.code} - $errorBody")
                    close(error)
                    return
                }

                response.body?.let { body ->
                    try {
                        val reader = body.byteStream().bufferedReader()

                        var line: String?
                        while (reader.readLine().also { line = it } != null) {
                            if (line?.startsWith("data: ") == true && !line!!.contains("data: [DONE]")) {
                                val content = line!!.substring(6)
                                try {
                                    val json = JSONObject(content)
                                    val choices = json.getJSONArray("choices")
                                    if (choices.length() > 0) {
                                        val choice = choices.getJSONObject(0)
                                        val delta = choice.getJSONObject("delta")

                                        if (delta.has("content")) {
                                            val text = delta.getString("content")

                                            if (isFirstToken) {
                                                trySend(Pair(text, 0))
                                                isFirstToken = false
                                            } else {
                                                trySend(Pair(text, 1))
                                            }
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(tag, "Error parsing JSON: $e")
                                }
                            }
                        }

                        reader.close()
                        close()
                    } catch (e: Exception) {
                        Log.e(tag, "Error reading response: $e")
                        close(e)
                    } finally {
                        body.close()
                    }
                } ?: close(IOException("Empty response body"))
            }
        })

        awaitClose {
            call.cancel()
        }
    }.flowOn(Dispatchers.IO)
}
