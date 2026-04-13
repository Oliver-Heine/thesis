package com.thesis.qrquishing.view

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.zxing.integration.android.IntentIntegrator
import com.google.zxing.integration.android.IntentResult
import com.thesis.qrquishing.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.TimeUnit
import kotlin.math.exp

/**
 * Main activity: scans a QR code, runs TFLite inference, and warns the user
 * if the decoded URL is classified as malicious or uncertain.
 */
class DebugActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "QRQuishing"
        private const val MAX_URL_LENGTH = 2048
        private const val BACKEND_URL = "http://10.0.2.2:8080/validate"  // localhost for emulator
    }

    // ── TFLite ──────────────────────────────────────────────────────────────
    private lateinit var tflite: Interpreter
    private lateinit var vocab: Map<String, Int>
    private var CONFIDENCE_THRESHOLD = 0.80f

    // ── HTTP client ─────────────────────────────────────────────────────────
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    // ── Coroutine scope tied to activity lifecycle ───────────────────────────
    private val activityScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // ── Views ────────────────────────────────────────────────────────────────
    private lateinit var tvUrl: TextView
    private lateinit var tvVerdict: TextView
    private lateinit var progressConfidence: ProgressBar
    private lateinit var tvConfidence: TextView
    private lateinit var settingsButton: FloatingActionButton
    private lateinit var urlInput: TextView

    private lateinit var cardResult: CardView
    private lateinit var bannerWarning: View

    // ── Camera permission launcher ───────────────────────────────────────────
    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startScan() else showPermissionDeniedDialog()
        }

    // ────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ────────────────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug)

        tvUrl = findViewById(R.id.tvUrl)
        tvVerdict = findViewById(R.id.tvVerdict)
        progressConfidence = findViewById(R.id.progressConfidence)
        tvConfidence = findViewById(R.id.tvConfidence)
        cardResult = findViewById(R.id.cardResult)
        bannerWarning = findViewById(R.id.bannerWarning)
        settingsButton = findViewById(R.id.btnSettings)
        urlInput = findViewById(R.id.urlInput)

        cardResult.visibility = View.GONE
        bannerWarning.visibility = View.GONE

        loadModel()
        loadVocab()

        intent.getStringExtra("CONFIDENCE_THRESHOLD")?.let { confidence ->
            CONFIDENCE_THRESHOLD = confidence.toFloat()
        }

        findViewById<View>(R.id.btnScan).setOnClickListener { handleScannedUrl(urlInput.text.toString()) }

        settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        activityScope.cancel()
        if (::tflite.isInitialized) tflite.close()
        httpClient.dispatcher.executorService.shutdown()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Model & vocab loading
    // ────────────────────────────────────────────────────────────────────────

    private fun loadModel() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "distilbert_model.tflite")
            val options = Interpreter.Options().apply { numThreads = 2 }
            tflite = Interpreter(modelBuffer, options)
            Log.i(TAG, "TFLite model loaded")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model", e)
            showErrorDialog("Model load failed", "Could not load the on-device model: ${e.message}")
        }
    }

    private fun loadVocab() {
        try {
            val mutableVocab = mutableMapOf<String, Int>()
            assets.open("vocab.txt").use { stream ->
                BufferedReader(InputStreamReader(stream))
                    .lineSequence()
                    .forEachIndexed { idx, line ->
                        mutableVocab[line.trim()] = idx
                    }
            }
            vocab = mutableVocab
            Log.i(TAG, "Vocab loaded: ${vocab.size} tokens")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocab", e)
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Camera permission & scanning
    // ────────────────────────────────────────────────────────────────────────

    private fun checkCameraAndScan() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED -> startScan()
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                AlertDialog.Builder(this)
                    .setTitle("Camera Required")
                    .setMessage("The camera is needed to scan QR codes.")
                    .setPositiveButton("Grant") { _, _ ->
                        requestCameraPermission.launch(Manifest.permission.CAMERA)
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
            else -> requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startScan() {
        val integrator = IntentIntegrator(this)
        integrator.setDesiredBarcodeFormats(IntentIntegrator.QR_CODE)
        integrator.setPrompt("Scan a QR code")
        integrator.setCameraId(0)
        integrator.setBeepEnabled(false)
        integrator.setBarcodeImageEnabled(false)
        integrator.initiateScan()
    }

    @Deprecated("Deprecated — required by ZXing IntentIntegrator")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val result: IntentResult? = IntentIntegrator.parseActivityResult(requestCode, resultCode, data)
        if (result != null) {
            val rawUrl = result.contents
            if (rawUrl.isNullOrBlank()) {
                showInfoDialog("No QR Code Found", "No QR code was detected. Please try again.")
            } else {
                handleScannedUrl(rawUrl)
            }
        } else {
            @Suppress("DEPRECATION")
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Inference pipeline
    // ────────────────────────────────────────────────────────────────────────

    private fun handleScannedUrl(rawUrl: String) {
        val url = rawUrl.trim().lowercase()

        // Basic sanity check before running inference
        if (url.length > MAX_URL_LENGTH) {
            showWarningDialog(rawUrl, "UNCERTAIN", 0f, "URL exceeds maximum length.")
            return
        }

        activityScope.launch {
            val (verdict, confidence) = withContext(Dispatchers.Default) {
                runLocalInference(url)
            }

            updateResultCard(url, verdict, confidence)

            when {
                verdict == "MALICIOUS" -> showWarningDialog(url, verdict, confidence, null)
                verdict == "UNCERTAIN" -> {
                    // Escalate to backend for uncertain cases
                    val backendResult = withContext(Dispatchers.IO) {
                        queryBackend(url)
                    }
                    if (backendResult != null) {
                        val backendVerdict = backendResult.optString("verdict", "uncertain").uppercase()
                        val backendConf = backendResult.optDouble("confidence", 0.0).toFloat()
                        updateResultCard(url, backendVerdict, backendConf)
                        if (backendVerdict == "MALICIOUS" || backendVerdict == "UNCERTAIN") {
                            showWarningDialog(url, backendVerdict, backendConf, null)
                        }
                    } else {
                        // Backend unreachable — surface the uncertainty
                        showWarningDialog(url, verdict, confidence, "Backend check failed; treat with caution.")
                    }
                }
                // BENIGN with high confidence — no warning needed
            }
        }
    }

    /** Tokenize the URL and run TFLite classification. */
    private fun runLocalInference(url: String): Pair<String, Float> {
        if (!::tflite.isInitialized || !::vocab.isInitialized) {
            Log.w(TAG, "Model or vocab not ready")
            return Pair("UNCERTAIN", 0f)
        }

        return try {
            val maxLen = 128
            val unknownId = vocab["[UNK]"] ?: 0
            val padId = vocab["[PAD]"] ?: 0
            val clsId = vocab["[CLS]"] ?: 101
            val sepId = vocab["[SEP]"] ?: 102

            // Character-level tokenization: the TFLite model exported from
            // train.py embeds a character vocabulary so that the full
            // WordPiece tokenizer is not needed on-device. Each character
            // maps to its vocab ID; unknown characters map to [UNK].
            val tokens = url.split("").filter { it.isNotEmpty() }.map { vocab[it] ?: unknownId }
            val ids = LongArray(maxLen) { padId.toLong() }
            val mask = LongArray(maxLen) { 0L }

            ids[0] = clsId.toLong()
            mask[0] = 1L
            val contentLen = minOf(tokens.size, maxLen - 2)
            for (i in 0 until contentLen) {
                ids[i + 1] = tokens[i].toLong()
                mask[i + 1] = 1L
            }
            val endIdx = contentLen + 1
            if (endIdx < maxLen) {
                ids[endIdx] = sepId.toLong()
                mask[endIdx] = 1L
            }

            val inputIds = arrayOf(ids)
            val attentionMask = arrayOf(mask)
            val outputBuffer = Array(1) { FloatArray(2) }

            tflite.runForMultipleInputsOutputs(
                arrayOf(inputIds, attentionMask),
                mapOf(0 to outputBuffer)
            )

            val logits = outputBuffer[0]
            val maxLogit = maxOf(logits[0], logits[1])
            val expBenign = exp((logits[0] - maxLogit).toDouble()).toFloat()
            val expMalicious = exp((logits[1] - maxLogit).toDouble()).toFloat()
            val sum = expBenign + expMalicious
            val pMalicious = expMalicious / sum

            val verdict = when {
                pMalicious >= com.thesis.qrquishing.utils.Settings.CONFIDENCE_THRESHOLD -> "MALICIOUS"
                (1f - pMalicious) >= com.thesis.qrquishing.utils.Settings.CONFIDENCE_THRESHOLD -> "BENIGN"
                else -> "UNCERTAIN"
            }
            Pair(verdict, pMalicious)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            Pair("UNCERTAIN", 0f)
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Backend call (for uncertain cases)
    // ────────────────────────────────────────────────────────────────────────

    private fun queryBackend(url: String): JSONObject? {
        return try {
            val body = JSONObject().put("url", url).toString()
                .toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
                .url(BACKEND_URL)
                .post(body)
                .build()
            httpClient.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    response.body?.string()?.let { JSONObject(it) }
                } else {
                    Log.w(TAG, "Backend returned HTTP ${response.code}")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Backend request failed", e)
            null
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // UI helpers
    // ────────────────────────────────────────────────────────────────────────

    private fun updateResultCard(url: String, verdict: String, confidence: Float) {
        cardResult.visibility = View.VISIBLE
        tvUrl.text = url

        tvVerdict.text = verdict
        tvVerdict.setTextColor(
            ContextCompat.getColor(
                this,
                when (verdict) {
                    "MALICIOUS" -> android.R.color.holo_red_dark
                    "BENIGN" -> android.R.color.holo_green_dark
                    else -> android.R.color.holo_orange_dark
                }
            )
        )

        val pct = (confidence * 100).toInt()
        progressConfidence.progress = pct
        tvConfidence.text = "Confidence: $pct %"

        bannerWarning.visibility = if (verdict == "MALICIOUS" || verdict == "UNCERTAIN") {
            View.VISIBLE
        } else {
            View.GONE
        }
    }

    private fun showWarningDialog(url: String, verdict: String, confidence: Float, extra: String?) {
        val message = buildString {
            append("URL: $url\n\n")
            append("Verdict: $verdict\n")
            append("Confidence: ${(confidence * 100).toInt()}%\n")
            if (!extra.isNullOrBlank()) append("\n$extra")
            append("\n\nDo NOT open this URL.")
        }
        AlertDialog.Builder(this)
            .setTitle("⚠ Warning")
            .setMessage(message)
            .setPositiveButton("Understood", null)
            .setCancelable(false)
            .show()
    }

    private fun showPermissionDeniedDialog() {
        AlertDialog.Builder(this)
            .setTitle("Permission Denied")
            .setMessage("Camera access is required to scan QR codes. Please grant the permission in Settings.")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showErrorDialog(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showInfoDialog(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }
}
