package com.deepfake.detector

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class MissingModelTest {
    @Test fun missingModelDoesNotCrashOrStartRecording() {
        val controller = Robolectric.buildService(AudioAnalysisService::class.java).create()
        val service = controller.get()
        assertFalse(service.isReady)
        assertFalse(service.isRecording.value)
        assertTrue(service.errorMessage.value!!.contains("model.tflite"))
        controller.destroy()
    }
}
