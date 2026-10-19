package com.example.llama

enum class CpuMeterMode {
    FREQUENCY,
    AUTO,
    SYSTEM,
    PROCESS,
    ;

    fun stored(): String = when (this) {
        FREQUENCY -> "frequency"
        AUTO -> "auto"
        SYSTEM -> "system"
        PROCESS -> "process"
    }

    companion object {
        fun fromStored(value: String?): CpuMeterMode = when (value) {
            "auto" -> AUTO
            "system" -> SYSTEM
            "process" -> PROCESS
            else -> FREQUENCY
        }
    }
}

enum class CpuSampleSource {
    SYSTEM,
    FREQUENCY,
    PROCESS,
}

data class CoreClock(val currentKhz: Long, val minKhz: Long, val maxKhz: Long)

data class CpuStatSnapshot(val total: Long, val idle: Long)

/**
 * `/proc/stat` 第一行。idle 含 iowait。total 取前 8 列。
 * 小米 14 上应用读得到；部分新机型会 Permission denied。
 */
fun parseCpuStatLine(line: String): CpuStatSnapshot? {
    if (!line.startsWith("cpu ")) return null
    val values = line.trim().split(Regex("\\s+")).drop(1).mapNotNull { it.toLongOrNull() }
    if (values.isEmpty()) return null
    val idle = (values.getOrNull(3) ?: 0L) + (values.getOrNull(4) ?: 0L)
    val total = values.take(minOf(8, values.size)).sum()
    if (total <= 0L) return null
    return CpuStatSnapshot(total, idle)
}

/**
 * CPU Stats 在 Android 8 之后读不到 /proc/stat，就用各核频率估算：
 * (当前频率 - 最低) / (最高 - 最低)，再按频率区间加权。
 * 当前频率为 0 的核视为休眠，不计入。
 */
fun frequencyCpuPercent(cores: List<CoreClock>): Float? {
    var used = 0.0
    var span = 0.0
    for (core in cores) {
        val width = core.maxKhz - core.minKhz
        if (core.currentKhz <= 0L || width <= 0L) continue
        val current = core.currentKhz.coerceIn(core.minKhz, core.maxKhz)
        used += (current - core.minKhz).toDouble()
        span += width.toDouble()
    }
    if (span <= 0.0) return null
    return (used / span * 100.0).coerceIn(0.0, 100.0).toFloat()
}

fun systemCpuPercent(before: CpuStatSnapshot, after: CpuStatSnapshot): Float? {
    val total = after.total - before.total
    val idle = after.idle - before.idle
    if (total <= 0L || idle < 0L || idle > total) return null
    return ((total - idle).toFloat() / total.toFloat() * 100f).coerceIn(0f, 100f)
}

/**
 * 本进程 CPU 占用 = 进程消耗的 CPU 时间 / 墙钟时间 / 在线核心数。
 * 6 个线程打满、机器有 8 核时，结果是 75%，不是 100%。
 * 墙钟里包含等云端的时间，这段时间进程几乎不占 CPU，平均值会被拉低。
 */
fun processCpuPercent(cpuTimeMs: Long, wallTimeMs: Long, cores: Int): Float? {
    val coresUsed = processCpuCores(cpuTimeMs, wallTimeMs) ?: return null
    if (cores <= 0) return null
    return (coresUsed / cores.toDouble() * 100.0).coerceIn(0.0, 100.0).toFloat()
}

fun processCpuCores(cpuTimeMs: Long, wallTimeMs: Long): Double? {
    if (wallTimeMs <= 0L || cpuTimeMs < 0L) return null
    return cpuTimeMs.toDouble() / wallTimeMs.toDouble()
}
