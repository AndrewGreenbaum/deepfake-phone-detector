package com.deepfake.detector

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.DataType
import kotlin.math.max

/**
 * TensorFlow Lite classifier for phone-quality deepfake audio detection.
 * Optimized for 8kHz audio typical of phone calls.
 */
class DeepfakeClassifier(context: Context) {

    companion object {
        private const val MODEL_FILE = "model.tflite"
        private const val N_MELS = PhoneFeatures.N_MELS
        private const val NUM_FRAMES = PhoneFeatures.NUM_FRAMES
    }

    private val interpreter: Interpreter
    private val features = PhoneFeatures()

    init {
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
        try {
            require(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1) {
                "Expected one input and one output tensor."
            }
            require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 1, 64, 94)) &&
                interpreter.getInputTensor(0).dataType() == DataType.FLOAT32) {
                "Expected float32 input shaped [1, 1, 64, 94]."
            }
            require(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1, 2)) &&
                interpreter.getOutputTensor(0).dataType() == DataType.FLOAT32) {
                "Expected float32 output shaped [1, 2]."
            }
        } catch (e: Exception) {
            interpreter.close()
            throw e
        }
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        return context.assets.openFd(MODEL_FILE).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { stream ->
                stream.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }
    }

    /**
     * Analyze audio samples and return detection result.
     * @param audioSamples PCM audio samples (normalized -1 to 1)
     * @return DetectionResult with prediction and confidence
     */
    fun analyze(audioSamples: FloatArray): DetectionResult {
        val melSpec = features.transform(audioSamples)
        val inputBuffer = ByteBuffer.allocateDirect(4 * N_MELS * NUM_FRAMES)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (value in melSpec) inputBuffer.putFloat(value)
        inputBuffer.rewind()

        // Output tensor [1, 2] - probabilities for [real, fake]
        val outputBuffer = ByteBuffer.allocateDirect(4 * 2)
        outputBuffer.order(ByteOrder.nativeOrder())

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Parse output
        outputBuffer.rewind()
        val realLogit = outputBuffer.float
        val fakeLogit = outputBuffer.float

        // Softmax
        val maxLogit = max(realLogit, fakeLogit)
        val realExp = kotlin.math.exp(realLogit - maxLogit)
        val fakeExp = kotlin.math.exp(fakeLogit - maxLogit)
        val sum = realExp + fakeExp

        val realProb = realExp / sum
        val fakeProb = fakeExp / sum

        return DetectionResult(
            isDeepfake = fakeProb > realProb,
            confidence = max(realProb, fakeProb),
            realProbability = realProb,
            fakeProbability = fakeProb
        )
    }

    fun close() {
        interpreter.close()
    }
}

data class DetectionResult(
    val isDeepfake: Boolean,
    val confidence: Float,
    val realProbability: Float,
    val fakeProbability: Float
)
