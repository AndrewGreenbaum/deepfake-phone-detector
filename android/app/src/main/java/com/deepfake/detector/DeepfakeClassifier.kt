package com.deepfake.detector

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.ln
import kotlin.math.max

/**
 * TensorFlow Lite classifier for phone-quality deepfake audio detection.
 * Optimized for 8kHz audio typical of phone calls.
 */
class DeepfakeClassifier(context: Context) {

    companion object {
        private const val MODEL_FILE = "model.tflite"
        private const val SAMPLE_RATE = 8000
        private const val N_MELS = 64
        private const val N_FFT = 512
        private const val HOP_LENGTH = 128
        private const val NUM_FRAMES = 94  // ~1.5 seconds at 8kHz
        private const val CHUNK_SAMPLES = SAMPLE_RATE * 3 / 2  // 1.5 seconds
    }

    private val interpreter: Interpreter
    private val melFilterbank: Array<FloatArray>

    init {
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
        melFilterbank = createMelFilterbank()
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    /**
     * Analyze audio samples and return detection result.
     * @param audioSamples PCM audio samples (normalized -1 to 1)
     * @return DetectionResult with prediction and confidence
     */
    fun analyze(audioSamples: FloatArray): DetectionResult {
        // Ensure we have enough samples
        val samples = if (audioSamples.size < CHUNK_SAMPLES) {
            audioSamples.copyOf(CHUNK_SAMPLES)
        } else {
            audioSamples.copyOfRange(0, CHUNK_SAMPLES)
        }

        // Compute mel spectrogram
        val melSpec = computeMelSpectrogram(samples)

        // Prepare input tensor [1, 1, N_MELS, NUM_FRAMES]
        val inputBuffer = ByteBuffer.allocateDirect(4 * N_MELS * NUM_FRAMES)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (mel in 0 until N_MELS) {
            for (frame in 0 until NUM_FRAMES) {
                inputBuffer.putFloat(melSpec[mel][frame])
            }
        }
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

    private fun computeMelSpectrogram(samples: FloatArray): Array<FloatArray> {
        val numFrames = (samples.size - N_FFT) / HOP_LENGTH + 1
        val actualFrames = minOf(numFrames, NUM_FRAMES)

        val melSpec = Array(N_MELS) { FloatArray(NUM_FRAMES) }

        // Compute STFT and mel spectrogram
        for (frame in 0 until actualFrames) {
            val start = frame * HOP_LENGTH
            val frameData = samples.copyOfRange(start, minOf(start + N_FFT, samples.size))

            // Apply Hann window
            val windowed = applyHannWindow(frameData)

            // Compute magnitude spectrum via FFT
            val spectrum = computeFFTMagnitude(windowed)

            // Apply mel filterbank
            for (mel in 0 until N_MELS) {
                var sum = 0f
                for (bin in spectrum.indices) {
                    sum += spectrum[bin] * melFilterbank[mel][bin]
                }
                // Log scale with floor
                melSpec[mel][frame] = ln(max(sum, 1e-10f))
            }
        }

        return melSpec
    }

    private fun applyHannWindow(frame: FloatArray): FloatArray {
        val windowed = FloatArray(N_FFT)
        for (i in frame.indices) {
            val window = 0.5f * (1 - kotlin.math.cos(2 * Math.PI * i / (frame.size - 1))).toFloat()
            windowed[i] = frame[i] * window
        }
        return windowed
    }

    private fun computeFFTMagnitude(frame: FloatArray): FloatArray {
        // Simple DFT for magnitude (real implementation would use FFT library)
        val numBins = N_FFT / 2 + 1
        val magnitude = FloatArray(numBins)

        for (k in 0 until numBins) {
            var real = 0f
            var imag = 0f
            for (n in frame.indices) {
                val angle = -2.0 * Math.PI * k * n / N_FFT
                real += frame[n] * kotlin.math.cos(angle).toFloat()
                imag += frame[n] * kotlin.math.sin(angle).toFloat()
            }
            magnitude[k] = kotlin.math.sqrt(real * real + imag * imag)
        }

        return magnitude
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val numBins = N_FFT / 2 + 1
        val filterbank = Array(N_MELS) { FloatArray(numBins) }

        // Phone bandwidth: 300-3400 Hz
        val fMin = 300f
        val fMax = 3400f

        // Convert to mel scale
        fun hzToMel(hz: Float) = 2595f * kotlin.math.log10(1f + hz / 700f)
        fun melToHz(mel: Float) = 700f * (kotlin.math.pow(10.0, mel / 2595.0).toFloat() - 1f)

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = FloatArray(N_MELS + 2) { i ->
            melToHz(melMin + i * (melMax - melMin) / (N_MELS + 1))
        }

        // Convert to FFT bin indices
        val binPoints = melPoints.map { hz ->
            ((N_FFT + 1) * hz / SAMPLE_RATE).toInt().coerceIn(0, numBins - 1)
        }

        // Create triangular filters
        for (m in 0 until N_MELS) {
            for (k in binPoints[m] until binPoints[m + 1]) {
                filterbank[m][k] = (k - binPoints[m]).toFloat() / (binPoints[m + 1] - binPoints[m])
            }
            for (k in binPoints[m + 1] until binPoints[m + 2]) {
                filterbank[m][k] = (binPoints[m + 2] - k).toFloat() / (binPoints[m + 2] - binPoints[m + 1])
            }
        }

        return filterbank
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
