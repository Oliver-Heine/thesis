package com.thesis.qrquishing.view

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.thesis.qrquishing.R
import com.thesis.qrquishing.benchmarkrunner.DatasetLoader
import com.thesis.qrquishing.benchmarkrunner.InferenceRunner
import com.thesis.qrquishing.benchmarkrunner.ResultExporter
import com.thesis.qrquishing.model.ai.ModelProvider
import kotlinx.coroutines.*

class BenchmarkActivity : AppCompatActivity() {

    private lateinit var btnStart: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var tvStatus: TextView

    private val benchmarkScope = CoroutineScope(Dispatchers.Default + Job())

    private val models = listOf(
        "albert-base-v2.tflite",
        "distilbert-base-uncased.tflite",
        "google_mobilebert-uncased.tflite",
        "huawei-noah_TinyBERT_General_4L_312D.tflite",
        "huawei-noah_TinyBERT_General_6L_768D.tflite"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_benchmark)

        btnStart = findViewById(R.id.btnStartBenchmark)
        progressBar = findViewById(R.id.progressBar)
        tvStatus = findViewById(R.id.tvStatus)

        btnStart.setOnClickListener {
            startBenchmarkAllModels()
        }
    }

    private fun startBenchmarkAllModels() {
        btnStart.isEnabled = false
        benchmarkScope.launch {
            try {
                val loader = DatasetLoader()
                val totalSamples = 10000
                val chunkSize = 500
                val warmupSamples = 100

                for (modelName in models) {
                    // Load model in isolation
                    withContext(Dispatchers.Main) {
                        tvStatus.text = "Loading model $modelName..."
                        progressBar.progress = 0
                        progressBar.max = totalSamples
                    }

                    val classifier = ModelProvider.create(this@BenchmarkActivity, modelName)
                    val runner = InferenceRunner(classifier)

                    val metrics = runner.runBenchmark(
                        context = this@BenchmarkActivity,
                        datasetLoader = loader,
                        chunkSize = chunkSize,
                        warmupSamples = warmupSamples
                    ) { chunkIndex, avgMs, totalProcessed ->
                        launch(Dispatchers.Main) {
                            progressBar.progress = totalProcessed
                            tvStatus.text = "Model $modelName | Chunk $chunkIndex | Avg: ${"%.2f".format(avgMs)} ms | Processed: $totalProcessed/$totalSamples"
                        }
                    }

                    // Export results per model
                    val exporter = ResultExporter(this@BenchmarkActivity)
                    val fileName = "benchmark_${modelName.replace(".tflite", "")}.csv"
                    val file = exporter.exportMetrics(metrics, modelName, fileName = fileName)

                    withContext(Dispatchers.Main) {
                        tvStatus.text = "✅ Model $modelName benchmark complete! Saved: ${file?.absolutePath ?: "failed"}"
                    }

                    // Clean up for next model
                    classifier.close()
                    System.gc() // force garbage collection to free model memory
                    delay(500) // small delay to stabilize system resources
                }

                withContext(Dispatchers.Main) {
                    btnStart.isEnabled = true
                    tvStatus.text = "✅ All models benchmarked."
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "❌ Benchmark failed: ${e.message}"
                    btnStart.isEnabled = true
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        benchmarkScope.cancel() // avoid leaks
    }
}