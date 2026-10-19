package com.example.llama

import android.os.Handler
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile

data class CpuReading(
    val percent: Float,
    val source: CpuSampleSource,
    val fellBack: Boolean,
)

data class CpuMark(
    val wallMs: Long,
    val processCpuMs: Long,
    val stat: CpuStatSnapshot?,
    val freqPercent: Float?,
)

data class CpuDelta(
    val source: CpuSampleSource,
    val numer: Long,
    val denom: Long,
    val fellBack: Boolean,
)

/**
 * 整机口径先读 /proc/stat。Android 8 之后普通应用经常被 SELinux 拒绝，
 * CPU Stats 这时改用各核 scaling_cur_freq 相对 cpuinfo_min/max 的比例。
 * 只有频率也读不到时才退回本进程 CPU 时间。
 */
class CpuMonitor {
    var mode: CpuMeterMode = CpuMeterMode.FREQUENCY

    private var isMonitoring = false
    private val handler = Handler(Looper.getMainLooper())
    private var monitoringRunnable: Runnable? = null
    private var onCpuUsageUpdate: ((CpuReading) -> Unit)? = null
    private val workerThread = android.os.HandlerThread("cpu-monitor").apply { start() }
    private val workerHandler = Handler(workerThread.looper)

    private var lastCpuTimeMs = 0L
    private var lastWallTimeMs = 0L
    private var hasProcessBaseline = false
    private var lastStat: CpuStatSnapshot? = null
    private var resolved: CpuSampleSource? = null
    private var fellBackFromSystem = false
    private var loggedFirstSample = false
    private val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    fun applyMode(next: CpuMeterMode) {
        mode = next
        resolved = null
        fellBackFromSystem = false
        lastStat = null
        hasProcessBaseline = false
        loggedFirstSample = false
    }

    fun startMonitoring(intervalMs: Long = 500, callback: (CpuReading) -> Unit) {
        if (isMonitoring) {
            stopMonitoring()
        }
        onCpuUsageUpdate = callback
        isMonitoring = true
        hasProcessBaseline = false
        lastStat = null
        loggedFirstSample = false

        monitoringRunnable = object : Runnable {
            override fun run() {
                if (!isMonitoring) return
                val sample = readInterval()
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
        hasProcessBaseline = false
        lastStat = null
    }

    fun mark(): CpuMark = CpuMark(
        wallMs = SystemClock.elapsedRealtime(),
        processCpuMs = Process.getElapsedCpuTime(),
        stat = if (wantsProcStat()) readSystemStat() else null,
        freqPercent = if (mode == CpuMeterMode.PROCESS) null else readFrequencyPercent(),
    )

    fun deltaSince(mark: CpuMark): CpuDelta? {
        val wall = SystemClock.elapsedRealtime() - mark.wallMs
        val cpu = Process.getElapsedCpuTime() - mark.processCpuMs
        if (wantsProcStat()) {
            val statNow = readSystemStat()
            if (mark.stat != null && statNow != null) {
                val total = statNow.total - mark.stat.total
                val idle = statNow.idle - mark.stat.idle
                if (total > 0L && idle >= 0L && idle <= total) {
                    if (mode == CpuMeterMode.AUTO) resolved = CpuSampleSource.SYSTEM
                    return CpuDelta(CpuSampleSource.SYSTEM, total - idle, total, fellBack = false)
                }
                return null
            }
            noteProcStatUnreadable()
        }
        if (mode != CpuMeterMode.PROCESS && resolved != CpuSampleSource.PROCESS) {
            val now = readFrequencyPercent()
            val percent = average(mark.freqPercent, now)
            if (percent != null && wall > 0L) {
                resolved = CpuSampleSource.FREQUENCY
                val scaled = Math.round(percent.toDouble() * wall.toDouble())
                return CpuDelta(CpuSampleSource.FREQUENCY, scaled, wall, fellBackFromSystem)
            }
        }
        if (mode != CpuMeterMode.PROCESS) {
            resolved = CpuSampleSource.PROCESS
            fellBackFromSystem = true
        }
        if (wall <= 0L || cpu < 0L) return null
        return CpuDelta(CpuSampleSource.PROCESS, cpu, wall, fellBackFromSystem)
    }

    private fun readInterval(): CpuReading? {
        if (wantsProcStat()) {
            val stat = readSystemStat()
            if (stat != null) {
                val previous = lastStat
                lastStat = stat
                if (previous == null) return null
                val usage = systemCpuPercent(previous, stat) ?: return null
                if (mode == CpuMeterMode.AUTO) resolved = CpuSampleSource.SYSTEM
                logFirst(usage, CpuSampleSource.SYSTEM)
                return CpuReading(usage, CpuSampleSource.SYSTEM, fellBack = false)
            }
            noteProcStatUnreadable()
            val freq = readFrequencyPercent()
            if (freq != null) {
                resolved = CpuSampleSource.FREQUENCY
                logFirst(freq, CpuSampleSource.FREQUENCY)
                return CpuReading(freq, CpuSampleSource.FREQUENCY, fellBack = true)
            }
            resolved = CpuSampleSource.PROCESS
        } else if (mode == CpuMeterMode.FREQUENCY || resolved == CpuSampleSource.FREQUENCY) {
            val freq = readFrequencyPercent()
            if (freq != null) {
                resolved = CpuSampleSource.FREQUENCY
                logFirst(freq, CpuSampleSource.FREQUENCY)
                return CpuReading(freq, CpuSampleSource.FREQUENCY, fellBackFromSystem)
            }
        }
        val usage = readProcessInterval() ?: return null
        logFirst(usage, CpuSampleSource.PROCESS)
        return CpuReading(usage, CpuSampleSource.PROCESS, fellBackFromSystem)
    }

    private fun wantsProcStat(): Boolean = when (mode) {
        CpuMeterMode.PROCESS, CpuMeterMode.FREQUENCY -> false
        CpuMeterMode.SYSTEM, CpuMeterMode.AUTO ->
            resolved == null || resolved == CpuSampleSource.SYSTEM
    }

    private fun readProcessInterval(): Float? {
        val cpuTimeMs = Process.getElapsedCpuTime()
        val wallTimeMs = SystemClock.elapsedRealtime()
        if (!hasProcessBaseline) {
            lastCpuTimeMs = cpuTimeMs
            lastWallTimeMs = wallTimeMs
            hasProcessBaseline = true
            return null
        }
        val cpuDelta = cpuTimeMs - lastCpuTimeMs
        val wallDelta = wallTimeMs - lastWallTimeMs
        lastCpuTimeMs = cpuTimeMs
        lastWallTimeMs = wallTimeMs
        return processCpuPercent(cpuDelta, wallDelta, cores)
    }

    private fun readSystemStat(): CpuStatSnapshot? {
        return try {
            RandomAccessFile("/proc/stat", "r").use { file ->
                val line = file.readLine() ?: return null
                parseCpuStatLine(line)
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun noteProcStatUnreadable() {
        val first = !fellBackFromSystem
        fellBackFromSystem = true
        if (first) Log.w("CpuMonitor", "/proc/stat unreadable, using core frequencies")
    }

    private fun readFrequencyPercent(): Float? {
        val cpuDir = File("/sys/devices/system/cpu")
        val cores = cpuDir.listFiles { file ->
            file.isDirectory && file.name.matches(Regex("cpu\\d+"))
        } ?: return null
        val clocks = cores.mapNotNull { core ->
            val current = readKhz(File(core, "cpufreq/scaling_cur_freq")) ?: return@mapNotNull null
            val max = readKhz(File(core, "cpufreq/cpuinfo_max_freq"))
                ?: readKhz(File(core, "cpufreq/scaling_max_freq"))
                ?: return@mapNotNull null
            val min = readKhz(File(core, "cpufreq/cpuinfo_min_freq"))
                ?: readKhz(File(core, "cpufreq/scaling_min_freq"))
                ?: 0L
            CoreClock(current, min, max)
        }
        return frequencyCpuPercent(clocks)
    }

    private fun readKhz(file: File): Long? {
        return try {
            FileInputStream(file).bufferedReader().use { reader ->
                reader.readLine()?.trim()?.toLongOrNull()
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun average(start: Float?, end: Float?): Float? = when {
        start != null && end != null -> (start + end) / 2f
        else -> end ?: start
    }

    private fun logFirst(usage: Float, source: CpuSampleSource) {
        if (loggedFirstSample) return
        loggedFirstSample = true
        Log.i("CpuMonitor", "cpu ${"%.1f".format(usage)}% source=$source mode=$mode")
    }
}
