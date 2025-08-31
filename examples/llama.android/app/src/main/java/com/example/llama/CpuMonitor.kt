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
import java.util.concurrent.TimeUnit
import java.io.File
import java.io.FileInputStream

/**
 * CPU利用率监控工具类
 * 适用于Android系统的CPU使用率监控
 */
class CpuMonitor {
    // 系统级 CPU 统计基线（来源：/proc/stat）
    private var lastTotalCpuTicks = 0L
    private var lastIdleCpuTicks = 0L
    private var hasCpuBaseline = false
    private var isMonitoring = false
    private val handler = Handler(Looper.getMainLooper())
    private var monitoringRunnable: Runnable? = null

    private var onCpuUsageUpdate: ((Float) -> Unit)? = null

    /**
     * 开始监控CPU使用率
     * @param intervalMs 监控间隔时间（毫秒）
     * @param callback CPU使用率更新回调，参数为CPU使用率百分比（0.0-100.0）
     */
    fun startMonitoring(intervalMs: Long = 100, callback: (Float) -> Unit) {
        if (isMonitoring) {
            stopMonitoring()
        }

        onCpuUsageUpdate = callback
        isMonitoring = true

        // 初始读取一次系统 CPU 统计，建立基准值
        readSystemCpuStat().let { (total, idle) ->
            if (total > 0L) {
                lastTotalCpuTicks = total
                lastIdleCpuTicks = idle
                hasCpuBaseline = true
            } else {
                hasCpuBaseline = false
            }
        }

        monitoringRunnable = object : Runnable {
            override fun run() {
                if (isMonitoring) {
                    val cpuUsage = calculateSystemCpuUsage()
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
     * 计算系统整体 CPU 使用率（依据 /proc/stat 的增量法）
     * @return CPU使用率百分比（0.0-100.0）
     */
    private fun calculateSystemCpuUsage(): Float {
        val (total, idle) = readSystemCpuStat()
        if (total > 0L && idle >= 0L) {
            if (!hasCpuBaseline) {
                lastTotalCpuTicks = total
                lastIdleCpuTicks = idle
                hasCpuBaseline = true
                return 0.0f
            }

            val totalDiff = total - lastTotalCpuTicks
            val idleDiff = idle - lastIdleCpuTicks

            lastTotalCpuTicks = total
            lastIdleCpuTicks = idle

            if (totalDiff > 0L) {
                val usage = (totalDiff - idleDiff).toFloat() / totalDiff.toFloat() * 100f
                Log.d("CpuMonitor", "System CPU Usage: $usage% (totalDiff=$totalDiff idleDiff=$idleDiff)")
                return usage.coerceIn(0f, 100f)
            }
        }

        // /proc/stat 不可用或计算无效，尝试使用 top 输出作为回退
        readCpuFromTop()?.let { return it }

        // 若 top 也不可用，最后回退到根据各核频率估算
        readCpuFromFrequencies()?.let { return it }

        Log.w("CpuMonitor", "Fallback to 0% CPU (no /proc/stat and top not available)")
        return 0.0f
    }

    private fun readCpuFromTop(): Float? {
        val candidates = listOf(
            listOf("top", "-n", "1", "-b"),
            listOf("top", "-n", "1")
        )
        for (args in candidates) {
            try {
                val proc = ProcessBuilder(args).redirectErrorStream(true).start()
                proc.inputStream.bufferedReader().use { reader ->
                    val lines = mutableListOf<String>()
                    var count = 0
                    while (count < 20) {
                        val line = reader.readLine() ?: break
                        lines += line
                        count += 1
                    }
                    // 等待片刻并终止进程
                    proc.waitFor(200, TimeUnit.MILLISECONDS)
                    if (proc.isAlive) proc.destroy()

                    // 尝试解析包含 CPU 汇总的行
                    for (line in lines) {
                        val parsed = parseTopCpuLine(line)
                        if (parsed != null) return parsed
                    }
                }
            } catch (e: Exception) {
                Log.w("CpuMonitor", "top invocation failed: ${e.message}")
            }
        }
        return null
    }

    private fun parseTopCpuLine(line: String): Float? {
        val s = line.trim()
        if (!s.contains("%", ignoreCase = true)) return null
        // 可能的样式1: "CPU: 7% user, 3% system, 0% iowait, 0% irq, 90% idle"
        Regex("CPU: \\s*([0-9.]+)%\\s*user,\\s*([0-9.]+)%\\s*system,.*?([0-9.]+)%\\s*idle", RegexOption.IGNORE_CASE)
            .find(s)?.let { m ->
                val idle = m.groupValues[3].toFloatOrNull() ?: return null
                return (100f - idle).coerceIn(0f, 100f)
            }
        // 可能的样式2: "Cpu(s): 7.0%us, 3.0%sy, 0.0%ni, 90.0%id, ..."
        Regex("Cpu\\(s\\): \\s*([0-9.]+)%us,\\s*([0-9.]+)%sy,.*?([0-9.]+)%id", RegexOption.IGNORE_CASE)
            .find(s)?.let { m ->
                val idle = m.groupValues[3].toFloatOrNull() ?: return null
                return (100f - idle).coerceIn(0f, 100f)
            }
        // 可能的样式3: 不含 idle，近似为非空闲之和（可能含 io/irq/steal）
        if (s.startsWith("CPU", ignoreCase = true)) {
            val percents = Regex("([0-9.]+)%").findAll(s).mapNotNull { it.groupValues[1].toFloatOrNull() }.toList()
            if (percents.isNotEmpty()) {
                // 若包含 idle，就按 100-idle；否则按总和且不超过100
                val idleIdx = s.indexOf("idle", ignoreCase = true)
                return if (idleIdx >= 0) {
                    null
                } else {
                    percents.sum().coerceAtMost(100f)
                }
            }
        }
        return null
    }

    // 第三层回退：根据各核当前频率与最大频率估算 CPU 占用（加权平均）
    private fun readCpuFromFrequencies(): Float? {
        try {
            val cpuDir = File("/sys/devices/system/cpu")
            if (!cpuDir.exists() || !cpuDir.isDirectory) return null
            val cpuDirs = cpuDir.listFiles { f -> f.isDirectory && f.name.matches(Regex("cpu\\d+")) }?.sortedBy { it.name } ?: return null
            var weightedSum = 0.0
            var totalWeight = 0.0
            for (core in cpuDirs) {
                val cur = readLongFirstToken(File(core, "cpufreq/scaling_cur_freq"))
                    ?: readLongFirstToken(File(core, "cpufreq/cpuinfo_cur_freq"))
                val max = readLongFirstToken(File(core, "cpufreq/cpuinfo_max_freq"))
                    ?: readLongFirstToken(File(core, "cpufreq/scaling_max_freq"))
                val min = readLongFirstToken(File(core, "cpufreq/cpuinfo_min_freq"))
                    ?: readLongFirstToken(File(core, "cpufreq/scaling_min_freq"))
                    ?: 0L
                if (cur != null && max != null) {
                    val span = (max - min).coerceAtLeast(0L)
                    if (span > 0L) {
                        val adjCur = cur.coerceIn(min, max)
                        val util = ((adjCur - min).toDouble() / span.toDouble()).coerceIn(0.0, 1.0)
                        // 使用 (max - min) 作为权重，更符合时钟区间的有效计算能力
                        weightedSum += util * span.toDouble()
                        totalWeight += span.toDouble()
                    }
                }
            }
            if (totalWeight > 0.0) {
                val usage = (weightedSum / totalWeight * 100.0).toFloat()
                Log.d("CpuMonitor", "CPU (freq-based) Usage: $usage%")
                return usage.coerceIn(0f, 100f)
            }
        } catch (e: Exception) {
            Log.w("CpuMonitor", "freq-based cpu usage failed: ${e.message}")
        }
        return null
    }

    private fun readLongFirstToken(file: File): Long? {
        return try {
            if (!file.exists()) return null
            FileInputStream(file).bufferedReader().use { br ->
                val text = br.readLine()?.trim() ?: return null
                text.split(Regex("\\s+")).firstOrNull()?.toLongOrNull()
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * 读取系统 CPU 统计（/proc/stat 第一行）：
     * 返回 Pair(totalTicks, idleTicks)
     */
    private fun readSystemCpuStat(): Pair<Long, Long> {
        return try {
            RandomAccessFile("/proc/stat", "r").use { file ->
                val line = file.readLine() ?: return Pair(-1L, -1L)
                if (!line.startsWith("cpu ")) return Pair(-1L, -1L)
                val parts = line.trim().split(Regex("\\s+")).drop(1)
                if (parts.isEmpty()) return Pair(-1L, -1L)
                // 参考字段顺序：user nice system idle iowait irq softirq steal guest guest_nice
                val values = parts.mapNotNull { it.toLongOrNull() }
                if (values.isEmpty()) return Pair(-1L, -1L)
                val idle = values.getOrNull(3) ?: 0L
                val iowait = values.getOrNull(4) ?: 0L
                val idleAll = idle + iowait
                val total = values.take(8.coerceAtMost(values.size)).sum()
                Pair(total, idleAll)
            }
        } catch (e: Exception) {
            Log.e("CpuMonitor", "Failed to read /proc/stat", e)
            Pair(-1L, -1L)
        }
    }
}
