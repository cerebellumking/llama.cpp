package com.example.llama

import android.os.Handler
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.Log

/**
 * 采样本进程的 CPU 时间。应用读不了 /proc/stat（这台手机上是 Permission denied），
 * 所以不能用整机 idle 差分；频率和 top 也不能代表这个进程。
 */
class CpuMonitor {
    private var isMonitoring = false
    private val handler = Handler(Looper.getMainLooper())
    private var monitoringRunnable: Runnable? = null
    private var onCpuUsageUpdate: ((Float) -> Unit)? = null
    private val workerThread = android.os.HandlerThread("cpu-monitor").apply { start() }
    private val workerHandler = Handler(workerThread.looper)

    private var lastCpuTimeMs = 0L
    private var lastWallTimeMs = 0L
    private var hasBaseline = false
    private var loggedFirstSample = false
    private val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    fun startMonitoring(intervalMs: Long = 500, callback: (Float) -> Unit) {
        if (isMonitoring) {
            stopMonitoring()
        }
        onCpuUsageUpdate = callback
        isMonitoring = true
        hasBaseline = false
        loggedFirstSample = false

        monitoringRunnable = object : Runnable {
            override fun run() {
                if (!isMonitoring) return
                val sample = readIntervalPercent()
                if (sample != null) {
                    handler.post { onCpuUsageUpdate?.invoke(sample) }
                }
                workerHandler.postDelayed(this, intervalMs)
            }
        }
        workerHandler.post(monitoringRunnable!!)
    }

    fun stopMonitoring() {
        isMonitoring = false
        monitoringRunnable?.let {
            workerHandler.removeCallbacks(it)
            handler.removeCallbacksAndMessages(null)
        }
        monitoringRunnable = null
        onCpuUsageUpdate = null
        hasBaseline = false
    }

    private fun readIntervalPercent(): Float? {
        val cpuTimeMs = Process.getElapsedCpuTime()
        val wallTimeMs = SystemClock.elapsedRealtime()
        if (!hasBaseline) {
            lastCpuTimeMs = cpuTimeMs
            lastWallTimeMs = wallTimeMs
            hasBaseline = true
            return null
        }
        val cpuDelta = cpuTimeMs - lastCpuTimeMs
        val wallDelta = wallTimeMs - lastWallTimeMs
        lastCpuTimeMs = cpuTimeMs
        lastWallTimeMs = wallTimeMs
        val usage = processCpuPercent(cpuDelta, wallDelta, cores) ?: return null
        if (!loggedFirstSample) {
            loggedFirstSample = true
            Log.i(
                "CpuMonitor",
                "process cpu ${"%.1f".format(usage)}% cores=$cores cpuMs=$cpuDelta wallMs=$wallDelta",
            )
        }
        return usage
    }
}
