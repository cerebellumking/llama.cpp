package com.example.llama

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
