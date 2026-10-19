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
