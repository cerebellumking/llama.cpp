package com.example.llama.api

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
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

class DeepseekApiService : ApiService {
    private val tag = "DeepseekApiService"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    companion object {
        private var instance: DeepseekApiService? = null
        
        fun getInstance(): DeepseekApiService {
            if (instance == null) {
                instance = DeepseekApiService()
            }
            return instance!!
        }

        private const val API_URL = "https://api.deepseek.com/chat/completions"
        private const val MODEL_NAME = "deepseek-chat"
        private const val API_KEY = "your-api-key-here"  // 这里填入写死的API key
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    override fun send(prompt: String): Flow<Pair<String, Int>> = callbackFlow {
        val jsonBody = JSONObject().apply {
            put("model", MODEL_NAME)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", "You are a helpful assistant.")
                })
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", prompt)
                })
            })
            put("stream", true)
        }

        val requestBody = jsonBody.toString().toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(API_URL)
            .addHeader("Content-Type", "application/json")
            .addHeader("Authorization", "Bearer $API_KEY")
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
                        trySend(Pair("\nDeepSeek API completed", 1))
                        close()
                    } catch (e: Exception) {
                        Log.e(tag, "Error reading response: $e")
                        trySend(Pair("\nDeepSeek API error: $e", 1))
                        close(e)
                    } finally {
                        body.close()
                    }
                } ?: run {
                    trySend(Pair("\nDeepSeek API error: Empty response body", 1))
                    close(IOException("Empty response body"))
                }
            }
        })

        awaitClose {
            call.cancel()
        }
    }.flowOn(Dispatchers.IO)
}
