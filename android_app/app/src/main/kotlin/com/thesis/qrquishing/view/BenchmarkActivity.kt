package com.thesis.qrquishing.view

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Spinner
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

    private lateinit var modelSpinner: Spinner

    private lateinit var selectedModel: String

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
        modelSpinner = findViewById(R.id.spinnerModels)

        modelSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, models)

        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                selectedModel = models[position]
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedModel = models[0]
            }
        }



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
                val warmupSamples = 1000

                // Load model in isolation
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Loading model $selectedModel..."
                    progressBar.progress = 0
                    progressBar.max = totalSamples
                }

                val classifier = ModelProvider.create(this@BenchmarkActivity, selectedModel)
                val runner = InferenceRunner(classifier)

                val metrics = runner.runBenchmark(
                    context = this@BenchmarkActivity,
                    datasetLoader = loader,
                    chunkSize = chunkSize,
                    warmupSamples = warmupSamples
                ) { chunkIndex, avgMs, totalProcessed ->
                    launch(Dispatchers.Main) {
                        progressBar.progress = totalProcessed
                        tvStatus.text = "Model $selectedModel | Chunk $chunkIndex | Avg: ${"%.2f".format(avgMs)} ms | Processed: $totalProcessed/$totalSamples"
                    }
                }

                // Export results per model
                val exporter = ResultExporter(this@BenchmarkActivity)
                val fileName = "benchmark_${selectedModel.replace(".tflite", "")}.csv"
                val file = exporter.exportMetrics(metrics, selectedModel, fileName = fileName)

                withContext(Dispatchers.Main) {
                    tvStatus.text = "✅ Model $selectedModel benchmark complete! Saved: ${file?.absolutePath ?: "failed"}"
                }

                // Clean up for next model
                classifier.close()
                System.gc() // force garbage collection to free model memory
                delay(500) // small delay to stabilize system resources


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