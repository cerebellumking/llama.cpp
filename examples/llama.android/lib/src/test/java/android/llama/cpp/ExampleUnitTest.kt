package android.llama.cpp

import org.junit.Test

import org.junit.Assert.*

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    @Test
    fun addition_isCorrect() {
        assertEquals(4, 2 + 2)
    }

    @Test
    fun draftStrategy_normalizesAdaptiveAndDefaultsToFixed3() {
        assertEquals("adaptive", LLamaAndroid.normalizeDraftStrategy("adaptive"))
        assertEquals("adaptive", LLamaAndroid.normalizeDraftStrategy("Adaptive"))
        assertEquals("fixed3", LLamaAndroid.normalizeDraftStrategy("fixed3"))
        assertEquals("fixed3", LLamaAndroid.normalizeDraftStrategy("both"))
        assertEquals("fixed3", LLamaAndroid.normalizeDraftStrategy(""))
    }
}
