package android.llama.cpp

import android.util.Log
import androidx.annotation.Keep
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    private val tag: String? = this::class.simpleName

    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("ai-chat")

            // Set llama log handler to Android
            log_to_android()
            backend_init(nativeLibraryDir)

            Log.d(tag, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private val nlen: Int = 512
    private var nativeStatePtr: Long = 0L
    @Volatile
    private var heteroServerUrl: String = DEFAULT_HETERO_SERVER_URL
    @Volatile
    private var heteroTargetModel: String = DEFAULT_HETERO_TARGET_MODEL
    @Volatile
    private var heteroTokenizerId: String = DEFAULT_HETERO_TOKENIZER_ID
    @Volatile
    private var draftStrategy: String = DRAFT_STRATEGY_FIXED3
    @Volatile
    private var useQueueInGoodput: Boolean = false

    @Volatile
    private var nativeLibraryDir: String = ""

    private external fun log_to_android()
    private external fun load_model(filename: String): Long
    private external fun free_model(model: Long)
    private external fun new_context(model: Long): Long
    private external fun free_context(context: Long)
    private external fun backend_init(nativeLibDir: String)
    private external fun backend_free()
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun free_batch(batch: Long)
    private external fun new_sampler(minicpmChat: Boolean): Long
    private external fun free_sampler(sampler: Long)

    private external fun system_info(): String

    private external fun completion_init(
        context: Long,
        batch: Long,
        text: String,
        formatChat: Boolean,
        nLen: Int
    ): Int

    private external fun completion_loop(
        context: Long,
        batch: Long,
        sampler: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    private external fun heterospec_init(
        context: Long,
        batch: Long,
        text: String,
        formatChat: Boolean,
        nLen: Int,
        serverUrl: String,
        draftStrategy: String,
        useQueueInGoodput: Boolean,
        targetModel: String,
        tokenizerId: String,
    ): Int

    private external fun heterospec_cleanup()

    private external fun heterospec_loop(
        context: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    private external fun kv_cache_clear(context: Long)

    suspend fun load(pathToModel: String) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    val model = load_model(pathToModel)
                    if (model == 0L)  throw IllegalStateException("load_model() failed")

                    val context = new_context(model)
                    if (context == 0L) throw IllegalStateException("new_context() failed")

                    val batch = new_batch(1536, 0, 1)
                    if (batch == 0L) throw IllegalStateException("new_batch() failed")

                    val minicpmChat = pathToModel.contains("MiniCPM5", ignoreCase = true)
                    val sampler = new_sampler(minicpmChat)
                    if (sampler == 0L) throw IllegalStateException("new_sampler() failed")

                    Log.i(tag, "Loaded model $pathToModel")
                    threadLocalState.set(State.Loaded(model, context, batch, sampler))
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    fun setNativeLibraryDir(dir: String) {
        nativeLibraryDir = dir.trim()
    }

    fun setHeteroServerUrl(url: String) {
        heteroServerUrl = url.trim().ifEmpty { DEFAULT_HETERO_SERVER_URL }
    }

    fun setHeteroIcnIdentity(targetModel: String, tokenizerId: String) {
        heteroTargetModel = targetModel.trim().ifEmpty { DEFAULT_HETERO_TARGET_MODEL }
        heteroTokenizerId = tokenizerId.trim().ifEmpty { "$heteroTargetModel-tokenizer" }
    }

    fun setDraftStrategy(strategy: String) {
        draftStrategy = normalizeDraftStrategy(strategy)
    }

    fun getDraftStrategy(): String = draftStrategy

    fun setUseQueueInGoodput(enabled: Boolean) {
        useQueueInGoodput = enabled
    }

    fun getUseQueueInGoodput(): Boolean = useQueueInGoodput

    fun sendHetero(
        message: String,
        formatChat: Boolean = false,
        serverUrl: String = heteroServerUrl,
        draftStrategy: String = this.draftStrategy,
        useQueueInGoodput: Boolean = this.useQueueInGoodput,
    ): Flow<Pair<String, Int>> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                val strategy = normalizeDraftStrategy(draftStrategy)
                Log.i(tag, "sendHetero draft_strategy=$strategy use_queue_in_goodput=$useQueueInGoodput target=$heteroTargetModel tokenizer=$heteroTokenizerId")
                val ncur = IntVar(
                    heterospec_init(
                        state.context,
                        state.batch,
                        message,
                        formatChat,
                        nlen,
                        serverUrl.trim().ifEmpty { DEFAULT_HETERO_SERVER_URL },
                        strategy,
                        useQueueInGoodput,
                        heteroTargetModel,
                        heteroTokenizerId,
                    )
                )
                var promptLen = ncur.value
                if(ncur.value < 0) {
                    Log.e(tag, "Heterospec init failed")
                    emit(Pair("Heterospec init failed", ncur.value))
                    return@flow
                }
                try {
                    var curTokens = ncur.value
                    while (ncur.value <= promptLen + nlen) {
                        currentCoroutineContext().ensureActive()
                        val before = ncur.value
                        val str = heterospec_loop(state.context, promptLen + nlen, ncur)
                        currentCoroutineContext().ensureActive()
                        Log.i(
                            tag,
                            "hetero_phone kotlin before=$before after=${ncur.value} text=${str?.take(120)}",
                        )
                        if (str == null || (str.isEmpty() && ncur.value == curTokens)) {
                            throw IllegalStateException("协同解码中断：云端没有继续返回 token")
                        }
                        if (str.contains("[eog]")) {
                            val finalText = str.replace("[eog]", "")
                            if (finalText.isNotEmpty()) {
                                val totalTokens = ncur.value - curTokens
                                emit(Pair(finalText, totalTokens))
                            }
                            break
                        }

                        val totalTokens = ncur.value - curTokens
                        curTokens = ncur.value
                        emit(Pair(str, totalTokens))
                    }
                } finally {
                    kv_cache_clear(state.context)
                    heterospec_cleanup()
                }
            }
            else -> {}
        }
    }.flowOn(runLoop)

    fun send(message: String, formatChat: Boolean = false): Flow<Pair<String, Int>> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                val ncur = IntVar(completion_init(state.context, state.batch, message, formatChat, nlen))
                var totalTokens = 0
                var curTokens = ncur.value
                try {
                    while (ncur.value <= nlen) {
                        currentCoroutineContext().ensureActive()
                        val str = completion_loop(state.context, state.batch, state.sampler, nlen, ncur)
                        if (str == null) {
                            break
                        }
                        totalTokens = ncur.value - curTokens
                        curTokens = ncur.value
                        emit(Pair(str, totalTokens))
                    }
                } finally {
                    kv_cache_clear(state.context)
                }
            }
            else -> {}
        }
    }.flowOn(runLoop)

    /**
     * Unloads the model and frees resources.
     *
     * This is a no-op if there's no model loaded.
     */
    suspend fun unload() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    heterospec_cleanup()
                    free_context(state.context)
                    free_model(state.model)
                    free_batch(state.batch)
                    free_sampler(state.sampler);

                    threadLocalState.set(State.Idle)
                }
                else -> {}
            }
        }
    }

    companion object {
        const val DEFAULT_HETERO_SERVER_URL = "ws://124.222.246.189:25079/ws/predict/binary/v3"
        const val DEFAULT_HETERO_TARGET_MODEL = "qwen3-4b"
        const val DEFAULT_HETERO_TOKENIZER_ID = "qwen3-4b-tokenizer"
        const val DRAFT_STRATEGY_FIXED3 = "fixed3"
        const val DRAFT_STRATEGY_ADAPTIVE = "adaptive"

        fun normalizeDraftStrategy(strategy: String): String {
            return if (strategy.trim().equals(DRAFT_STRATEGY_ADAPTIVE, ignoreCase = true)) {
                DRAFT_STRATEGY_ADAPTIVE
            } else {
                DRAFT_STRATEGY_FIXED3
            }
        }

        // JNI (completion_loop / heterospec_loop) looks up getValue() and inc() by name.
        // Keep the class public and unstripped: R8 otherwise deletes inc() because Kotlin
        // never calls it, then JNI abort: NoSuchMethodError IntVar.inc()V.
        @Keep
        class IntVar(value: Int) {
            @Volatile
            var value: Int = value
                private set

            @Keep
            @JvmName("inc")
            fun inc() {
                synchronized(this) {
                    value += 1
                }
            }
        }

        @Keep
        @JvmField
        @Suppress("unused")
        val jniKeepIntVarInc: (IntVar) -> Unit = IntVar::inc

        private sealed interface State {
            data object Idle: State
            data class Loaded(val model: Long, val context: Long, val batch: Long, val sampler: Long): State
        }

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()

        fun instance(): LLamaAndroid = _instance
    }
}
