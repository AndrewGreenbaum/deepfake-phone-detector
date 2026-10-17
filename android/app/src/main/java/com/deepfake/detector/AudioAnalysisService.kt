package com.deepfake.detector

import android.app.*
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Foreground service for continuous audio analysis.
 * Records audio at 8kHz (phone quality) and runs deepfake detection.
 */
class AudioAnalysisService : Service() {

    companion object {
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "deepfake_detection"
        private const val SAMPLE_RATE = 8000
        private const val CHUNK_DURATION_MS = 3000  // Match the Python training window
        private const val CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION_MS / 1000
    }

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage
    val isReady: Boolean get() = classifier != null

    private val binder = LocalBinder()
    private var audioRecord: AudioRecord? = null
    private var classifier: DeepfakeClassifier? = null
    private val classifierLock = Any()
    private var analysisJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording

    private val _currentResult = MutableStateFlow<DetectionResult?>(null)
    val currentResult: StateFlow<DetectionResult?> = _currentResult

    private val _audioLevel = MutableStateFlow(0f)
    val audioLevel: StateFlow<Float> = _audioLevel

    inner class LocalBinder : Binder() {
        fun getService(): AudioAnalysisService = this@AudioAnalysisService
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        try {
            classifier = DeepfakeClassifier(this)
        } catch (e: Exception) {
            _errorMessage.value = "Detector unavailable: add a compatible trained model.tflite " +
                "to app/src/main/assets and rebuild. ${e.message ?: "Model could not be loaded."}"
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!startRecording()) stopSelf()
        return START_NOT_STICKY
    }

    private fun startRecording(): Boolean {
        if (_isRecording.value) return true
        if (!isReady) return false
        _errorMessage.value = null
        try {
            val notification = createNotification()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            val bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            check(bufferSize > 0) { "This device does not support 8 kHz microphone recording." }
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                maxOf(bufferSize, CHUNK_SIZE * 2)
            )
            audioRecord = recorder
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "Microphone could not be initialized." }
            recorder.startRecording()
            check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "Microphone did not start." }
            _isRecording.value = true

            analysisJob = scope.launch {
                val buffer = ShortArray(CHUNK_SIZE)
                var filled = 0
                try {
                    while (isActive && _isRecording.value) {
                        val read = recorder.read(buffer, filled, CHUNK_SIZE - filled)
                        check(read >= 0) { "Microphone read failed ($read)." }
                        if (read == 0) {
                            delay(10)
                            continue
                        }
                        filled += read
                        if (filled < CHUNK_SIZE) continue
                        val samples = FloatArray(CHUNK_SIZE) { buffer[it] / 32768f }
                        val level = samples.maxOf { kotlin.math.abs(it) }
                        _audioLevel.value = level
                        if (level > 0.01f) {
                            val result = synchronized(classifierLock) {
                                if (isActive && _isRecording.value) classifier?.analyze(samples) else null
                            }
                            if (isActive && _isRecording.value) _currentResult.value = result
                        }
                        filled = 0
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    if (isActive && _isRecording.value) {
                        _errorMessage.value = "Detection stopped: ${e.message}"
                        withContext(Dispatchers.Main) { stopRecording() }
                    }
                }
            }
            return true
        } catch (e: Exception) {
            _errorMessage.value = "Unable to start microphone: ${e.message}"
            stopRecording()
            return false
        }
    }

    fun stopRecording() {
        _isRecording.value = false
        analysisJob?.cancel()
        analysisJob = null
        audioRecord?.let { recorder ->
            try {
                if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop()
            } finally {
                recorder.release()
            }
        }
        audioRecord = null
        _currentResult.value = null
        _audioLevel.value = 0f
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Deepfake detection notifications"
                setShowBadge(false)
            }

            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    override fun onDestroy() {
        stopRecording()
        scope.cancel()
        synchronized(classifierLock) {
            classifier?.close()
            classifier = null
        }
        super.onDestroy()
    }
}
