package com.deepfake.detector

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.deepfake.detector.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var audioService: AudioAnalysisService? = null
    private var serviceBound = false

    private val recentResults = mutableListOf<DetectionResult>()

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as AudioAnalysisService.LocalBinder
            audioService = localBinder.getService()
            serviceBound = true
            observeService()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            audioService = null
            serviceBound = false
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            startDetection()
        } else {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        bindAudioService()
    }

    private fun setupUI() {
        binding.startButton.setOnClickListener {
            if (audioService?.isRecording?.value == true) {
                stopDetection()
            } else {
                checkPermissionsAndStart()
            }
        }

        updateUI(isRecording = false, result = null)
    }

    private fun bindAudioService() {
        Intent(this, AudioAnalysisService::class.java).also { intent ->
            startService(intent)
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
    }

    private fun observeService() {
        lifecycleScope.launch {
            audioService?.isRecording?.collectLatest { isRecording ->
                updateUI(isRecording, audioService?.currentResult?.value)
            }
        }

        lifecycleScope.launch {
            audioService?.currentResult?.collectLatest { result ->
                result?.let {
                    updateResultUI(it)
                    addToHistory(it)
                }
            }
        }

        lifecycleScope.launch {
            audioService?.audioLevel?.collectLatest { level ->
                updateWaveform(level)
            }
        }
    }

    private fun checkPermissionsAndStart() {
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (notGranted.isEmpty()) {
            startDetection()
        } else {
            permissionLauncher.launch(notGranted.toTypedArray())
        }
    }

    private fun startDetection() {
        audioService?.startRecording()
        updateUI(isRecording = true, result = null)
    }

    private fun stopDetection() {
        audioService?.stopRecording()
        updateUI(isRecording = false, result = null)
    }

    private fun updateUI(isRecording: Boolean, result: DetectionResult?) {
        binding.startButton.text = if (isRecording) {
            getString(R.string.stop_detection)
        } else {
            getString(R.string.start_detection)
        }

        binding.statusText.text = when {
            isRecording && result == null -> getString(R.string.status_listening)
            isRecording -> getString(R.string.status_analyzing)
            else -> getString(R.string.status_idle)
        }

        if (!isRecording) {
            binding.resultText.text = "--"
            binding.resultText.setTextColor(getColor(R.color.text_primary))
            binding.confidenceText.text = ""
        }
    }

    private fun updateResultUI(result: DetectionResult) {
        val (text, color) = if (result.isDeepfake) {
            "FAKE" to getColor(R.color.result_fake)
        } else {
            "REAL" to getColor(R.color.result_real)
        }

        binding.resultText.text = text
        binding.resultText.setTextColor(color)
        binding.confidenceText.text = getString(
            R.string.confidence,
            result.confidence * 100
        )

        // Pulse animation on detection
        binding.resultCard.animate()
            .scaleX(1.05f)
            .scaleY(1.05f)
            .setDuration(100)
            .withEndAction {
                binding.resultCard.animate()
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(100)
                    .start()
            }
            .start()
    }

    private fun updateWaveform(level: Float) {
        // Simple visualization - scale the view based on audio level
        val scale = 0.3f + (level * 2f).coerceAtMost(0.7f)
        binding.waveformView.scaleY = scale
    }

    private fun addToHistory(result: DetectionResult) {
        recentResults.add(0, result)
        if (recentResults.size > 5) {
            recentResults.removeAt(recentResults.size - 1)
        }

        val fakeCount = recentResults.count { it.isDeepfake }
        val realCount = recentResults.size - fakeCount

        binding.historyText.text = "Recent: $realCount real, $fakeCount fake detections"
    }

    override fun onDestroy() {
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
        super.onDestroy()
    }
}
