package com.deepfake.detector

import kotlin.math.*

/** Deterministic Python evaluation preprocessing for mono 8 kHz PCM. */
class PhoneFeatures {
    companion object {
        const val SAMPLE_RATE = 8000
        const val NUM_SAMPLES = 24000
        const val N_MELS = 64
        const val NUM_FRAMES = 94
        private const val N_FFT = 512
        private const val HOP = 256
        private const val PAD = 27
        // scipy.signal.butter(4, [300, 3400], btype="band", fs=8000)
        private val B = doubleArrayOf(0.38783095426643766, 0.0, -1.5513238170657506, 0.0, 2.326985725598626, 0.0, -1.5513238170657506, 0.0, 0.38783095426643766)
        private val A = doubleArrayOf(1.0, -0.6219294583801217, -1.995147252969811, 0.8194174578198945, 1.8846887328385828, -0.4649270424761891, -0.8439998746981743, 0.0932978587392206, 0.1504644595333187)
        private val ZI = doubleArrayOf(-0.3878309542664326, -0.38783095426643577, 1.1634928627993046, 1.1634928627993089, -1.1634928627993075, -1.16349286279931, 0.3878309542664364, 0.3878309542664369)
    }

    private val window = FloatArray(N_FFT) { (0.5 * (1 - cos(2 * PI * it / N_FFT))).toFloat() }
    private val cosine = Array(N_FFT / 2 + 1) { k -> DoubleArray(N_FFT) { n -> cos(2 * PI * k * n / N_FFT) } }
    private val sine = Array(N_FFT / 2 + 1) { k -> DoubleArray(N_FFT) { n -> sin(2 * PI * k * n / N_FFT) } }
    private val filters = createFilters()

    fun transform(audio: FloatArray): FloatArray {
        require(audio.size > PAD) { "At least 28 audio samples are required." }
        val filtered = bandpass(audio)
        val peak = filtered.maxOf { abs(it) }
        val samples = FloatArray(NUM_SAMPLES) { i ->
            if (i < filtered.size && peak > 0) filtered[i] / peak * 0.95f else 0f
        }
        val result = FloatArray(N_MELS * NUM_FRAMES)
        val frame = FloatArray(N_FFT)
        val power = DoubleArray(N_FFT / 2 + 1)
        for (t in 0 until NUM_FRAMES) {
            for (n in 0 until N_FFT) {
                var index = t * HOP + n - N_FFT / 2
                if (index < 0) index = -index
                if (index >= samples.size) index = 2 * samples.size - index - 2
                frame[n] = samples[index] * window[n]
            }
            for (k in power.indices) {
                var real = 0.0
                var imaginary = 0.0
                for (n in frame.indices) {
                    real += frame[n] * cosine[k][n]
                    imaginary -= frame[n] * sine[k][n]
                }
                power[k] = real * real + imaginary * imaginary
            }
            for (m in 0 until N_MELS) {
                var energy = 0.0
                for (k in power.indices) energy += power[k] * filters[m][k]
                result[m * NUM_FRAMES + t] = (10 * log10(max(energy, 1e-10))).toFloat()
            }
        }
        val floor = result.maxOrNull()!! - 80f
        for (i in result.indices) result[i] = max(result[i], floor)
        return result
    }

    private fun bandpass(audio: FloatArray): FloatArray {
        val extended = DoubleArray(audio.size + 2 * PAD)
        for (i in 0 until PAD) {
            extended[i] = 2.0 * audio[0] - audio[PAD - i]
            extended[PAD + audio.size + i] = 2.0 * audio.last() - audio[audio.size - 2 - i]
        }
        for (i in audio.indices) extended[PAD + i] = audio[i].toDouble()
        val forward = filter(extended)
        forward.reverse()
        val backward = filter(forward)
        backward.reverse()
        return FloatArray(audio.size) { backward[it + PAD].toFloat() }
    }

    private fun filter(audio: DoubleArray): DoubleArray {
        val state = DoubleArray(ZI.size) { ZI[it] * audio[0] }
        return DoubleArray(audio.size) { i ->
            val output = B[0] * audio[i] + state[0]
            for (j in 0 until state.lastIndex) {
                state[j] = B[j + 1] * audio[i] + state[j + 1] - A[j + 1] * output
            }
            state[state.lastIndex] = B.last() * audio[i] - A.last() * output
            output
        }
    }

    private fun createFilters(): Array<FloatArray> {
        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        val low = hzToMel(300.0)
        val high = hzToMel(3400.0)
        val points = FloatArray(N_MELS + 2) { i ->
            (700.0 * (10.0.pow((low + (high - low) * i / (N_MELS + 1)) / 2595.0) - 1)).toFloat()
        }
        return Array(N_MELS) { m ->
            FloatArray(N_FFT / 2 + 1) { k ->
                val frequency = k.toFloat() * SAMPLE_RATE / N_FFT
                max(0f, min((frequency - points[m]) / (points[m + 1] - points[m]),
                            (points[m + 2] - frequency) / (points[m + 2] - points[m + 1])))
            }
        }
    }
}
