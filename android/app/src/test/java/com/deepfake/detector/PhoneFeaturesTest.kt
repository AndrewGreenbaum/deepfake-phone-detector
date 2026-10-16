package com.deepfake.detector

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhoneFeaturesTest {
    private fun fixture(name: String): FloatArray {
        val bytes = javaClass.getResourceAsStream("/phone_features/$name")!!.use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { buffer.float }
    }

    @Test fun matchesPythonEvaluationFeatures() {
        val transform = PhoneFeatures()
        for (name in listOf("silence", "sweep", "short_noise", "long_tones")) {
            val actual = transform.transform(fixture("$name.pcm.f32"))
            val expected = fixture("$name.mel.f32")
            assertEquals(expected.size, actual.size)
            val maximumError = actual.indices.maxOf { abs(actual[it] - expected[it]) }
            assertTrue("$name differs from Python by $maximumError dB", maximumError < 0.05f)
        }
    }
}
