package com.example.llama.api

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.EOFException
import java.io.IOException
import java.util.concurrent.TimeUnit

class QwenApiService : ApiService {
    private val tag = "QwenApiService"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(180, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    companion object {
        private var instance: QwenApiService? = null

        fun getInstance(): QwenApiService {
            if (instance == null) {
                instance = QwenApiService()
            }
            return instance!!
        }

        // Qwen3.8-27B-FP8 model card, thinking mode (the default).
        // https://huggingface.co/Qwen/Qwen3.8-27B-FP8
        const val QWEN38_TEMPERATURE = 1.0
        const val QWEN38_TOP_P = 0.95
        const val QWEN38_TOP_K = 20
        const val QWEN38_MIN_P = 0.0
        const val QWEN38_PRESENCE_PENALTY = 0.0
        const val QWEN38_REPETITION_PENALTY = 1.0
        const val QWEN38_MAX_NEW_TOKENS = 4096
    }

    @Volatile
    private var endpoint: String = ""

    @Volatile
    private var qwen38Sampling: Boolean = false

    fun configure(wsUrl: String, qwen38: Boolean) {
        qwen38Sampling = qwen38
        endpoint = generateUrlFromWs(wsUrl)
    }

    fun hasEndpoint(): Boolean = endpoint.isNotBlank()

    private fun generateUrlFromWs(wsUrl: String): String {
        val uri = android.net.Uri.parse(wsUrl.trim())
        val scheme = if (uri.scheme.equals("wss", ignoreCase = true)) "https" else "http"
        val host = uri.host ?: return ""
        val port = uri.port
        val authority = if (port != -1) "$host:$port" else host
        return "$scheme://$authority/generate"
    }

    override fun send(prompt: String): Flow<Pair<String, Int>> = send(prompt, maxNewTokens = null)

    @OptIn(ExperimentalCoroutinesApi::class)
    fun send(prompt: String, maxNewTokens: Int?): Flow<Pair<String, Int>> = callbackFlow {
        if (endpoint.isEmpty()) {
            close(IOException("Qwen API 地址为空"))
            awaitClose { }
            return@callbackFlow
        }

        val jsonBody = JSONObject().apply {
            put("prompt", prompt)
            put("stream", true)
            val limit = maxNewTokens ?: if (qwen38Sampling) QWEN38_MAX_NEW_TOKENS else 256
            put("max_new_tokens", limit)
            put("max_length", limit)
            if (qwen38Sampling) {
                put("temperature", QWEN38_TEMPERATURE)
                put("top_p", QWEN38_TOP_P)
                put("top_k", QWEN38_TOP_K)
                put("min_p", QWEN38_MIN_P)
                put("presence_penalty", QWEN38_PRESENCE_PENALTY)
                put("repetition_penalty", QWEN38_REPETITION_PENALTY)
            }
        }

        val requestBody = jsonBody.toString().toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(endpoint)
            .addHeader("Content-Type", "application/json")
            .post(requestBody)
            .build()

        val call = client.newCall(request)

        call.enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                Log.e(tag, "API request failed: ${e.message}", e)
                close(e)
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                if (!response.isSuccessful) {
                    val errorBody = response.body?.string() ?: "Unknown error"
                    Log.e(tag, "API error: ${response.code} - $errorBody")
                    close(IOException("API error: ${response.code} - $errorBody"))
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
                                            val content = delta.getString("content")
                                            val parts = content.split("�")
                                            if (parts.size >= 2) {
                                                val text = parts[0]
                                                val tokenCount = parts[1].trim().toInt()
                                                trySend(Pair(text, tokenCount))
                                            }
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(tag, "Error parsing response line: $line", e)
                                }
                            }
                        }

                        reader.close()
                        trySend(Pair("\nQwen API completed", 0))
                        close()
                    } catch (e: Exception) {
                        Log.e(tag, "Error processing response: $e")
                        trySend(Pair("\nQwen API error: $e", 1))
                        close(e)
                    } finally {
                        body.close()
                    }
                } ?: run {
                    trySend(Pair("Qwen API error: Empty response body", 1))
                    close(IOException("Empty response body"))
                }
            }
        })

        awaitClose {
            call.cancel()
        }
    }.flowOn(Dispatchers.IO)
}
