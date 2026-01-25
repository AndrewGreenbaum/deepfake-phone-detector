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
        private const val CHUNK_DURATION_MS = 1500  // 1.5 second chunks
        private const val CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION_MS / 1000
    }

    private val binder = LocalBinder()
    private var audioRecord: AudioRecord? = null
    private var classifier: DeepfakeClassifier? = null
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
        classifier = DeepfakeClassifier(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = createNotification()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        return START_STICKY
    }

    fun startRecording() {
        if (_isRecording.value) return

        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(bufferSize, CHUNK_SIZE * 2)
        )

        audioRecord?.startRecording()
        _isRecording.value = true

        analysisJob = scope.launch {
            val buffer = ShortArray(CHUNK_SIZE)
            val floatBuffer = FloatArray(CHUNK_SIZE)

            while (isActive && _isRecording.value) {
                val read = audioRecord?.read(buffer, 0, CHUNK_SIZE) ?: 0

                if (read > 0) {
                    // Convert to float and normalize
                    var maxAmp = 0f
                    for (i in 0 until read) {
                        floatBuffer[i] = buffer[i] / 32768f
                        maxAmp = maxOf(maxAmp, kotlin.math.abs(floatBuffer[i]))
                    }
                    _audioLevel.value = maxAmp

                    // Run detection if audio level is sufficient
                    if (maxAmp > 0.01f) {
                        try {
                            val result = classifier?.analyze(floatBuffer)
                            result?.let { _currentResult.value = it }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }
                }

                delay(100)  // Small delay between chunks
            }
        }
    }

    fun stopRecording() {
        _isRecording.value = false
        analysisJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        _currentResult.value = null
        _audioLevel.value = 0f
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
        classifier?.close()
        scope.cancel()
        super.onDestroy()
    }
}
