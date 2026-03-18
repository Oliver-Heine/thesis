package com.thesis.qrquishing

import android.Manifest
import android.app.AlertDialog
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
import com.google.zxing.integration.android.IntentIntegrator
import com.google.zxing.integration.android.IntentResult
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
import java.net.URI
import java.util.concurrent.TimeUnit
import kotlin.math.PI
import kotlin.math.exp

private const val MALICIOUS = "MALICIOUS"

private const val BENIGN = "BENIGN"

private const val UNCERTAIN = "UNCERTAIN"

/**
 * Main activity: scans a QR code, runs TFLite inference, and warns the user
 * if the decoded URL is classified as malicious or uncertain.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "QRQuishing"
        private const val CONFIDENCE_THRESHOLD = 0.80f
        private const val MAX_URL_LENGTH = 2048
        private const val BACKEND_URL = "http://10.0.2.2:8080/validate"
        private const val MODEL_SEQUENCE_LENGTH = 128
    }

    private lateinit var tflite: Interpreter
    private lateinit var vocab: Map<String, Int>
    private lateinit var idToToken: Map<Int, String>
    private var tokenizer: WordPieceTokenizer? = null

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val activityScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private lateinit var tvUrl: TextView
    private lateinit var tvVerdict: TextView
    private lateinit var progressConfidence: ProgressBar
    private lateinit var tvConfidence: TextView
    private lateinit var cardResult: CardView
    private lateinit var bannerWarning: View

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startScan() else showPermissionDeniedDialog()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvUrl = findViewById(R.id.tvUrl)
        tvVerdict = findViewById(R.id.tvVerdict)
        progressConfidence = findViewById(R.id.progressConfidence)
        tvConfidence = findViewById(R.id.tvConfidence)
        cardResult = findViewById(R.id.cardResult)
        bannerWarning = findViewById(R.id.bannerWarning)

        cardResult.visibility = View.GONE
        bannerWarning.visibility = View.GONE

        loadModel()
        loadVocab()

        findViewById<View>(R.id.btnScan).setOnClickListener { checkCameraAndScan() }
    }

    override fun onDestroy() {
        super.onDestroy()
        activityScope.cancel()
        if (::tflite.isInitialized) tflite.close()
        httpClient.dispatcher.executorService.shutdown()
    }

    private fun loadModel() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "distilbert_model.tflite")
            val options = Interpreter.Options().apply { numThreads = 2 }
            tflite = Interpreter(modelBuffer, options)

            // Resize inputs to match the 128 sequence length used in runLocalInference
            val inputShape = intArrayOf(1, MODEL_SEQUENCE_LENGTH)
            for (i in 0 until tflite.inputTensorCount) {
                tflite.resizeInput(i, inputShape)
            }
            tflite.allocateTensors()

            Log.i(TAG, "TFLite model loaded and resized to $MODEL_SEQUENCE_LENGTH")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model", e)
        }

        for (i in 0 until tflite.inputTensorCount) {
            val t = tflite.getInputTensor(i)
            Log.i(TAG, "Input[$i] name=${t.name()} shape=${t.shape().contentToString()} dtype=${t.dataType()}")
        }
        for (i in 0 until tflite.outputTensorCount) {
            val t = tflite.getOutputTensor(i)
            Log.i(TAG, "Output[$i] name=${t.name()} shape=${t.shape().contentToString()} dtype=${t.dataType()}")
        }
    }

    private fun loadVocab() {
        try {
            val mutableVocab = mutableMapOf<String, Int>()
            assets.open("vocab.txt").use { stream ->
                BufferedReader(InputStreamReader(stream))
                    .lineSequence()
                    .forEachIndexed { idx, line ->
                        val token = line.trim()
                        if (token.isNotEmpty()) {
                            mutableVocab[token] = idx
                        }
                    }
            }

            val required = listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<", ">")
            val missing = required.filter { !mutableVocab.containsKey(it) }
            if (missing.isNotEmpty()) {
                throw IllegalStateException("vocab.txt is missing required tokens: $missing")
            }

            vocab = mutableVocab
            idToToken = mutableVocab.entries.associate { it.value to it.key }
            tokenizer = WordPieceTokenizer(vocab)

            Log.i(TAG, "Vocab loaded: ${vocab.size} tokens")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocab", e)
        }
    }

    private fun checkCameraAndScan() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startScan()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startScan() {
        val integrator = IntentIntegrator(this)
        integrator.setDesiredBarcodeFormats(IntentIntegrator.QR_CODE)
        integrator.setPrompt("Scan a QR code")
        integrator.initiateScan()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: android.content.Intent?) {
        val result: IntentResult? = IntentIntegrator.parseActivityResult(requestCode, resultCode, data)
        if (result != null && !result.contents.isNullOrBlank()) {
            handleScannedUrl(result.contents)
        } else {
            @Suppress("DEPRECATION")
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    private fun handleScannedUrl(rawUrl: String) {
        val url = rawUrl.trim()
        if (url.isEmpty()) return

        activityScope.launch {
            val (verdict, confidence) = withContext(Dispatchers.Default) {
                runLocalInference(url)
            }

            updateResultCard(url, verdict, confidence)

            if (verdict == "MALICIOUS" || verdict == UNCERTAIN) {
                showWarningDialog(url, verdict, confidence, null)
            }
        }
    }

    /**
     * Normalizes URL to match the requested structure:
     * "< tag > value < tag > value ..."
     */
    private fun normalizeUrl(url: String): String {
        var cleanUrl = url.lowercase().trim()
        cleanUrl = cleanUrl.replace(Regex("^https?://"), "")
        cleanUrl = cleanUrl.replace(Regex("^www\\."), "")

        return try {
            val uri = URI("http://$cleanUrl")
            val host = uri.host ?: ""
            val hostParts = host.split(".").filter { it.isNotEmpty() }

            val subdomainParts = if (hostParts.size > 2) hostParts.dropLast(2) else emptyList()
            val domainPart = if (hostParts.size >= 2) hostParts[hostParts.size - 2]
            else hostParts.getOrNull(0) ?: ""
            val suffixParts = if (hostParts.size >= 2) listOf(hostParts.last()) else emptyList()

            val tokens = mutableListOf<String>()

            if (subdomainParts.isNotEmpty()) {
                tokens.add("<")
                tokens.add("subdomain")
                tokens.add(">")
                tokens.addAll(subdomainParts)
            }

            if (domainPart.isNotEmpty()) {
                tokens.add("<")
                tokens.add("domain")
                tokens.add(">")
                tokens.add(domainPart)
            }

            if (suffixParts.isNotEmpty()) {
                tokens.add("<")
                tokens.add("suffix")
                tokens.add(">")
                tokens.addAll(suffixParts)
            }

            uri.path?.takeIf { it != "/" }?.let { path ->
                val pathTokens = path.split(Regex("[/\\-_.?=&]")).filter { it.isNotEmpty() }
                if (pathTokens.isNotEmpty()) {
                    tokens.add("<")
                    tokens.add("path")
                    tokens.add(">")
                    tokens.addAll(pathTokens)
                }
            }

            uri.query?.takeIf { it.isNotBlank() }?.let { query ->
                val queryTokens = query.split(Regex("[=&]")).filter { it.isNotEmpty() }
                if (queryTokens.isNotEmpty()) {
                    tokens.add("<")
                    tokens.add("query")
                    tokens.add(">")
                    tokens.addAll(queryTokens)
                }
            }

            tokens.joinToString(" ")
        } catch (e: Exception) {
            cleanUrl
        }
    }

    private fun runLocalInference(rawUrl: String): Pair<String, Float> {
        val localTokenizer = tokenizer
        if (!::tflite.isInitialized || localTokenizer == null) {
            return Pair(UNCERTAIN, 0f)
        }

        return try {
            val normalized = normalizeUrl(rawUrl)

            val maxLen = MODEL_SEQUENCE_LENGTH
            val clsId = vocab["[CLS]"]?.toLong() ?: 101L
            val sepId = vocab["[SEP]"]?.toLong() ?: 102L
            val padId = vocab["[PAD]"]?.toLong() ?: 0L

            val wordPieceTokens = localTokenizer.tokenize(normalized)
            
            // Print tokens to terminal as requested
            Log.d(TAG, "${clsId}: [CLS]")
            wordPieceTokens.forEach { id ->
                Log.d(TAG, "${id}: ${idToToken[id]}")
            }
            Log.d(TAG, "${sepId}: [SEP]")

            val ids = LongArray(maxLen) { padId }
            val mask = LongArray(maxLen) { 0L }

            ids[0] = clsId
            mask[0] = 1L

            val contentLen = minOf(wordPieceTokens.size, maxLen - 2)
            for (i in 0 until contentLen) {
                ids[i + 1] = wordPieceTokens[i].toLong()
                mask[i + 1] = 1L
            }

            val endIdx = contentLen + 1
            if (endIdx < maxLen) {
                ids[endIdx] = sepId
                mask[endIdx] = 1L
            }

            val inputIds = Array(1) { ids }
            val attentionMask = Array(1) { mask }
            val outputBuffer = Array(1) { FloatArray(2) }

            val inputs = arrayOfNulls<Any>(tflite.inputTensorCount)

            for (i in 0 until tflite.inputTensorCount) {
                val tensor = tflite.getInputTensor(i)
                val name = tensor.name().lowercase()

                when {
                    "input_ids" in name -> inputs[i] = inputIds
                    "attention_mask" in name -> inputs[i] = attentionMask
                    else -> throw IllegalStateException(
                        "Unexpected input tensor[$i]: ${tensor.name()}"
                    )
                }
            }

            tflite.runForMultipleInputsOutputs(
                inputs.requireNoNulls(),
                mapOf(0 to outputBuffer)
            )

            val logits = outputBuffer[0]
            val maxLogit = maxOf(logits[0], logits[1])
            val expBenign = exp((logits[0] - maxLogit).toDouble()).toFloat()
            val expMalicious = exp((logits[1] - maxLogit).toDouble()).toFloat()
            val sum = expBenign + expMalicious
            val pMalicious = expMalicious / sum
            val pBenign = expBenign / sum

            
            if (pMalicious >= CONFIDENCE_THRESHOLD) {
                Pair(MALICIOUS, pMalicious)
            } else if ((1f - pMalicious) >= CONFIDENCE_THRESHOLD) {
                Pair(BENIGN, pBenign)
            } else {
                Pair(UNCERTAIN, pMalicious)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            Pair(UNCERTAIN, 0f)
        }
    }

    private fun queryBackend(url: String): JSONObject? {
        return try {
            val body = JSONObject().put("url", url).toString().toRequestBody("application/json".toMediaType())
            val request = Request.Builder().url(BACKEND_URL).post(body).build()
            httpClient.newCall(request).execute().use { response ->
                if (response.isSuccessful) response.body?.string()?.let { JSONObject(it) } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun updateResultCard(url: String, verdict: String, confidence: Float) {
        cardResult.visibility = View.VISIBLE
        tvUrl.text = url
        tvVerdict.text = verdict
        tvVerdict.setTextColor(ContextCompat.getColor(this, when (verdict) {
            "MALICIOUS" -> android.R.color.holo_red_dark
            BENIGN -> android.R.color.holo_green_dark
            else -> android.R.color.holo_orange_dark
        }))
        val pct = (confidence * 100).toInt()
        progressConfidence.progress = pct
        tvConfidence.text = "Confidence: $pct %"
        bannerWarning.visibility = if (verdict == "MALICIOUS" || verdict == UNCERTAIN) View.VISIBLE else View.GONE
    }

    private fun showWarningDialog(url: String, verdict: String, confidence: Float, extra: String?) {
        val message = "URL: $url\n\nVerdict: $verdict\nConfidence: ${(confidence * 100).toInt()}%\n\nDo NOT open this URL."
        AlertDialog.Builder(this).setTitle("⚠ Warning").setMessage(message).setPositiveButton("OK", null).show()
    }

    private fun showPermissionDeniedDialog() {
        AlertDialog.Builder(this).setTitle("Permission Denied").setMessage("Camera access is required.").setPositiveButton("OK", null).show()
    }

    /**
     * Standard WordPiece tokenizer. 
     */
    class WordPieceTokenizer(private val vocab: Map<String, Int>) {
        private val unknownId = vocab["[UNK]"] ?: 100
        
        fun tokenize(text: String): List<Int> {
            val result = mutableListOf<Int>()
            // Split by space first
            for (word in text.split(Regex("\\s+"))) {
                if (word.isEmpty()) continue
                
                var start = 0
                while (start < word.length) {
                    var end = word.length
                    var curToken = -1
                    
                    while (start < end) {
                        val subword = if (start == 0) word.substring(start, end) 
                                     else "##" + word.substring(start, end)
                        
                        if (vocab.containsKey(subword)) {
                            curToken = vocab[subword]!!
                            break
                        }
                        end--
                    }
                    
                    if (curToken == -1) {
                        result.add(unknownId)
                        break
                    } else {
                        result.add(curToken)
                        start = end
                    }
                }
            }
            return result
        }
    }
}
