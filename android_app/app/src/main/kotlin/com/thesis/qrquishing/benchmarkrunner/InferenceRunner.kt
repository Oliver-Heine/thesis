package com.thesis.qrquishing.benchmarkrunner

import android.content.Context
import android.util.Log
import com.thesis.qrquishing.benchmarkrunner.dto.InferenceMetrics
import com.thesis.qrquishing.model.ai.TFLiteClassifier

class InferenceRunner(
    private val classifier: TFLiteClassifier
) {

    /**
     * Runs the benchmark and reports progress to the UI.
     *
     * @param context Android context
     * @param datasetLoader The DatasetLoader instance
     * @param chunkSize How many samples per chunk
     * @param warmupSamples Number of warm-up samples
     * @param onChunkCompleted Callback for UI reporting per chunk
     */
    fun runBenchmark(
        context: Context,
        datasetLoader: DatasetLoader,
        chunkSize: Int = 10_000,
        warmupSamples: Int = 1000,
        onChunkCompleted: ((chunkIndex: Int, chunkAvgMs: Double, totalProcessed: Int) -> Unit)? = null
    ): InferenceMetrics {
        Log.d("Benchmark", "Starting benchmark")

        val metricsCollector = MetricsCollector(context)

        // 🔥 Warm-up phase
        warmup(context, datasetLoader, warmupSamples)

        var totalProcessed = 0

        val batteryMahStart = metricsCollector.getBatteryDischargeMah()
        datasetLoader.streamDatasetInChunks(
            context = context,
            chunkSize = chunkSize
        ) { chunk, chunkIndex ->
            val chunkStart = System.nanoTime()

            val chunkLatencies = mutableListOf<Double>()

            Log.d("Benchmark", "Running actual load")
            for (sample in chunk) {
                val start = System.nanoTime()
                classifier.classify(sample.url)
                val end = System.nanoTime()

                val latencyMs = (end - start) / 1_000_000.0
                chunkLatencies.add(latencyMs)
            }
            val chunkEnd = System.nanoTime()
            val chunkDurationMs = (chunkEnd - chunkStart) / 1_000_000.0

            metricsCollector.addChunk(chunkLatencies, chunkIndex, chunkDurationMs)
            totalProcessed += chunk.size

            // Call UI callback
            onChunkCompleted?.invoke(
                chunkIndex,
                chunkLatencies.average(),
                totalProcessed
            )

            Log.d("Benchmark", "Chunk $chunkIndex → avg=${"%.2f".format(chunkLatencies.average())} ms")
        }
        val batteryMahEnd = metricsCollector.getBatteryDischargeMah()
        val batteryMah = batteryMahStart - batteryMahEnd
        val batteryMahPerInference = batteryMah / chunkSize

        return metricsCollector.computeMetrics(batteryMah, batteryMahPerInference)
    }

    private fun warmup(
        context: Context,
        datasetLoader: DatasetLoader,
        warmupSamples: Int
    ) {
        var count = 0
        var stop = false
        Log.d("Benchmark", "Starting warmup")

        datasetLoader.streamDatasetInChunks(context, chunkSize = warmupSamples) { chunk, _ ->
            if (stop) return@streamDatasetInChunks

            for (sample in chunk) {
                val (verdict, confidence) = classifier.classify(sample.url)
                Log.d("Benchmark Verdict", String.format("URL %s got verdict: %s, with confidence: %f", sample, verdict.name, confidence))
                count++

                if (count >= warmupSamples) {
                    Log.d("Benchmark", "Warm-up completed ($count samples)")
                    stop = true
                    break
                }
            }
        }
    }
}