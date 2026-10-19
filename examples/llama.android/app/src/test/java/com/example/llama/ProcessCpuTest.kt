package com.example.llama

import com.example.llama.ui.eval.DATASET_STRATEGY_ADAPTIVE
import com.example.llama.ui.eval.DATASET_STRATEGY_BOTH
import com.example.llama.ui.eval.DATASET_STRATEGY_FIXED3
import com.example.llama.ui.eval.datasetTestPasses
import com.example.llama.ui.eval.visibleDatasets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ProcessCpuTest {
    @Test
    fun sixBusyThreadsOnEightCores_is75Percent() {
        val percent = processCpuPercent(cpuTimeMs = 6_000, wallTimeMs = 1_000, cores = 8)
        assertEquals(75f, percent!!, 0.01f)
        assertEquals(6.0, processCpuCores(6_000, 1_000)!!, 0.001)
    }

    @Test
    fun idleProcess_isZero() {
        assertEquals(0f, processCpuPercent(0, 500, 8)!!, 0.01f)
    }

    @Test
    fun invalidWindow_isNull() {
        assertNull(processCpuPercent(10, 0, 8))
        assertNull(processCpuPercent(-1, 500, 8))
        assertNull(processCpuPercent(10, 500, 0))
    }

    @Test
    fun oversubscribedProcess_clampsAt100() {
        assertEquals(100f, processCpuPercent(20_000, 1_000, 8)!!, 0.01f)
    }

    @Test
    fun procStat_usesIdlePlusIoWaitOverEightFields() {
        val before = parseCpuStatLine("cpu  100 0 50 800 50 10 20 5")
        val after = parseCpuStatLine("cpu  200 0 100 900 50 10 20 5 extra")
        assertEquals(CpuStatSnapshot(1035, 850), before)
        assertEquals(CpuStatSnapshot(1285, 950), after)
        assertEquals(60f, systemCpuPercent(before!!, after!!)!!, 0.01f)
    }

    @Test
    fun procStat_rejectsBlankAndNonCpuLines() {
        assertNull(parseCpuStatLine("cpu0 1 2 3 4"))
        assertNull(parseCpuStatLine(""))
        assertNull(systemCpuPercent(CpuStatSnapshot(10, 1), CpuStatSnapshot(10, 1)))
    }

    @Test
    fun frequencyFallback_weightsEachCoreByItsRange() {
        val percent = frequencyCpuPercent(
            listOf(
                CoreClock(currentKhz = 1_000, minKhz = 0, maxKhz = 1_000),
                CoreClock(currentKhz = 0, minKhz = 0, maxKhz = 2_000),
                CoreClock(currentKhz = 500, minKhz = 0, maxKhz = 1_000),
            ),
        )
        assertEquals(75f, percent!!, 0.01f)
    }

    @Test
    fun meterMode_roundTripsStoredValue() {
        assertEquals(CpuMeterMode.FREQUENCY, CpuMeterMode.fromStored(null))
        assertEquals(CpuMeterMode.FREQUENCY, CpuMeterMode.fromStored("frequency"))
        assertEquals(CpuMeterMode.AUTO, CpuMeterMode.fromStored("auto"))
        assertEquals(CpuMeterMode.SYSTEM, CpuMeterMode.fromStored("system"))
        assertEquals(CpuMeterMode.PROCESS, CpuMeterMode.fromStored("process"))
        assertEquals("frequency", CpuMeterMode.FREQUENCY.stored())
    }
}

class DatasetTestPlanTest {
    @Test
    fun localAndApi_runOnceAndDoNotTouchDraftStrategy() {
        assertEquals(listOf(null), datasetTestPasses(InferenceMode.LOCAL, DATASET_STRATEGY_BOTH))
        assertEquals(listOf(null), datasetTestPasses(InferenceMode.API, DATASET_STRATEGY_ADAPTIVE))
    }

    @Test
    fun hetero_honorsFixedAdaptiveAndBoth() {
        assertEquals(listOf(DATASET_STRATEGY_FIXED3), datasetTestPasses(InferenceMode.HETERO, "fixed3"))
        assertEquals(listOf(DATASET_STRATEGY_ADAPTIVE), datasetTestPasses(InferenceMode.HETERO, "Adaptive"))
        assertEquals(
            listOf(DATASET_STRATEGY_FIXED3, DATASET_STRATEGY_ADAPTIVE),
            datasetTestPasses(InferenceMode.HETERO, DATASET_STRATEGY_BOTH),
        )
        assertEquals(listOf(DATASET_STRATEGY_FIXED3), datasetTestPasses(InferenceMode.HETERO, "nope"))
    }

    @Test
    fun hiddenFiles_areNotDatasets() {
        assertEquals(listOf("reports", "test"), visibleDatasets(arrayOf(".DS_Store", "test", "reports", "")))
    }
}
