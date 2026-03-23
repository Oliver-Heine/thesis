package com.thesis.qrquishing.benchmarkrunner

import android.content.Context
import android.util.Log
import com.thesis.qrquishing.benchmarkrunner.dto.InferenceMetrics
import java.io.File
import java.io.FileWriter
import java.io.IOException

class ResultExporter(private val context: Context) {

    fun exportMetrics(metrics: InferenceMetrics, modelName: String, fileName: String = "_benchmark_results.csv"): File? {
        return try {
            val file = File(context.getExternalFilesDir(null), fileName)
            val writer = FileWriter(file)

            writer.apply {
                appendLine("Summary for $modelName")
                appendLine("Total Samples,${metrics.totalSamples}")
                appendLine("Avg Latency (ms),${metrics.avgLatencyMs}")
                appendLine("P50 Latency (ms),${metrics.p50LatencyMs}")
                appendLine("P95 Latency (ms),${metrics.p95LatencyMs}")
                appendLine("P99 Latency (ms),${metrics.p99LatencyMs}")
                appendLine("Min Latency (ms),${metrics.minLatencyMs}")
                appendLine("Max Latency (ms),${metrics.maxLatencyMs}")
                appendLine("")

                appendLine("ChunkIndex,AvgLatencyMs,SampleCount,AvgCpuPercent,AvgBatteryCurrentUa")
                metrics.chunkStats.forEach { chunk ->
                    appendLine(
                        "${chunk.chunkIndex},${chunk.avgLatencyMs},${chunk.sampleCount}," +
                                "${chunk.avgCpuPercent},${chunk.avgBatteryCurrentUa}"
                    )
                }

                flush()
                close()
            }

            Log.d("ResultExporter", "Metrics exported to ${file.absolutePath}")
            file

        } catch (e: IOException) {
            Log.e("ResultExporter", "Failed to export metrics", e)
            null
        }
    }
}