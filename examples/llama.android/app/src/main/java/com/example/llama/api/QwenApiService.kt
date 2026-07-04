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
        .readTimeout(30, TimeUnit.SECONDS)
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

        private const val API_URL = ""
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    override fun send(prompt: String): Flow<Pair<String, Int>> = callbackFlow {
        val jsonBody = JSONObject().apply {
            put("prompt", prompt)
            put("max_length", 256)  // 可以根据需要调整
            put("stream", true)
        }

        val requestBody = jsonBody.toString().toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(API_URL)
            .addHeader("Content-Type", "application/json")
            .post(requestBody)
            .build()

        var isFirstToken = true
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

                                                if (isFirstToken) {
                                                    trySend(Pair(text, 0))
                                                    isFirstToken = false
                                                } else {
                                                    trySend(Pair(text, tokenCount))
                                                }
                                            }
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(tag, "Error parsing response line: $line", e)
                                }
                            }
                        }

                        reader.close()
                        trySend(Pair("\nQwen API completed", 1))
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
