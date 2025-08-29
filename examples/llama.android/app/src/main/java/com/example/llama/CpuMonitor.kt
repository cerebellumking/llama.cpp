package com.example.llama

import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.IOException
import java.io.RandomAccessFile

/**
 * CPU利用率监控工具类
 * 适用于Android系统的CPU使用率监控
 */
class CpuMonitor {
    private var lastAppCpuTime = 0L
    private var lastSystemTime = 0L
    private var isMonitoring = false
    private val handler = Handler(Looper.getMainLooper())
    private var monitoringRunnable: Runnable? = null

    private var onCpuUsageUpdate: ((Float) -> Unit)? = null

    /**
     * 开始监控CPU使用率
     * @param intervalMs 监控间隔时间（毫秒）
     * @param callback CPU使用率更新回调，参数为CPU使用率百分比（0.0-100.0）
     */
    fun startMonitoring(intervalMs: Long = 50, callback: (Float) -> Unit) {
        if (isMonitoring) {
            stopMonitoring()
        }

        onCpuUsageUpdate = callback
        isMonitoring = true

        // 初始读取一次，建立基准值
        lastAppCpuTime = getAppCpuTime()
        lastSystemTime = System.nanoTime()

        monitoringRunnable = object : Runnable {
            override fun run() {
                if (isMonitoring) {
                    val cpuUsage = calculateAppCpuUsage()
                    onCpuUsageUpdate?.invoke(cpuUsage)
                    handler.postDelayed(this, intervalMs)
                }
            }
        }

        handler.postDelayed(monitoringRunnable!!, intervalMs)
    }

    /**
     * 停止监控CPU使用率
     */
    fun stopMonitoring() {
        isMonitoring = false
        monitoringRunnable?.let { handler.removeCallbacks(it) }
        monitoringRunnable = null
        onCpuUsageUpdate = null
    }

    /**
     * 计算应用CPU使用率
     * @return CPU使用率百分比（0.0-100.0）
     */
    private fun calculateAppCpuUsage(): Float {
        val currentAppCpuTime = getAppCpuTime()
        val currentSystemTime = System.nanoTime()

        if (currentAppCpuTime == -1L) {
            Log.w("CpuMonitor", "Failed to get app CPU time")
            return 0.0f
        }

        // 计算时间差
        val appCpuTimeDiff = currentAppCpuTime - lastAppCpuTime
        val systemTimeDiff = currentSystemTime - lastSystemTime

        Log.d("CpuMonitor", "appCpuTimeDiff: $appCpuTimeDiff, systemTimeDiff: $systemTimeDiff")

        // 更新上次的值
        lastAppCpuTime = currentAppCpuTime
        lastSystemTime = currentSystemTime

        // 计算CPU使用率 (将纳秒转换为百分比)
        val cpuUsage = if (systemTimeDiff > 0) {
            (appCpuTimeDiff.toFloat() / systemTimeDiff.toFloat()) * 100f
        } else {
            0.0f
        }

        Log.d("CpuMonitor", "App CPU Usage: $cpuUsage%")
        return cpuUsage.coerceIn(0f, 100f) // 确保结果在0-100范围内
    }

    /**
     * 获取当前应用的CPU时间 (纳秒)
     * @return CPU时间，失败时返回-1
     */
    private fun getAppCpuTime(): Long {
        return try {
            // 使用Android系统提供的API获取线程CPU时间
            Debug.threadCpuTimeNanos()
        } catch (e: Exception) {
            Log.e("CpuMonitor", "Failed to get thread CPU time", e)
            try {
                // 备用方法：读取/proc/self/stat
                val pid = Process.myPid()
                RandomAccessFile("/proc/$pid/stat", "r").use { file ->
                    val line = file.readLine()
                    val parts = line.split(" ")
                    if (parts.size >= 15) {
                        // utime (user time) + stime (system time)
                        val utime = parts[13].toLongOrNull() ?: 0L
                        val stime = parts[14].toLongOrNull() ?: 0L
                        // 转换为纳秒 (假设时钟频率为100Hz)
                        (utime + stime) * 10_000_000L
                    } else {
                        -1L
                    }
                }
            } catch (e2: Exception) {
                Log.e("CpuMonitor", "Failed to read /proc/self/stat", e2)
                -1L
            }
        }
    }
}
