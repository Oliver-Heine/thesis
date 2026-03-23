package com.thesis.qrquishing.benchmarkrunner

import android.content.Context
import android.os.BatteryManager
import com.thesis.qrquishing.benchmarkrunner.dto.ChunkStats
import com.thesis.qrquishing.benchmarkrunner.dto.InferenceMetrics
import java.io.RandomAccessFile

class MetricsCollector(private val context: Context) {

    private val latencies = mutableListOf<Double>()
    private val chunkStats = mutableListOf<ChunkStats>()
    private var totalSamples = 0
    private var lastCpuTotal: Long = 0
    private var lastCpuIdle: Long = 0

    /** Add a batch/chunk of latency measurements */
    fun addChunk(latenciesChunk: List<Double>, chunkIndex: Int) {
        if (latenciesChunk.isEmpty()) return

        latencies.addAll(latenciesChunk)
        totalSamples += latenciesChunk.size

        val avgLatency = latenciesChunk.average()
        val avgCpu = getCpuUsagePercent()
        val avgBatteryCurrent = getBatteryCurrentUa()

        chunkStats.add(
            ChunkStats(
                chunkIndex = chunkIndex,
                avgLatencyMs = avgLatency,
                sampleCount = latenciesChunk.size,
                avgCpuPercent = avgCpu,
                avgBatteryCurrentUa = avgBatteryCurrent
            )
        )
    }

    /** Add a single latency measurement */
    fun add(latencyMs: Double) {
        latencies.add(latencyMs)
        totalSamples++
    }

    /** Compute final metrics */
    fun computeMetrics(): InferenceMetrics {
        val sorted = latencies.sorted()
        fun percentile(p: Double): Double {
            val index = (p * sorted.size).toInt().coerceAtMost(sorted.lastIndex)
            return sorted[index]
        }

        return InferenceMetrics(
            totalSamples = totalSamples,
            avgLatencyMs = latencies.average(),
            p50LatencyMs = percentile(0.50),
            p95LatencyMs = percentile(0.95),
            p99LatencyMs = percentile(0.99),
            minLatencyMs = sorted.firstOrNull() ?: 0.0,
            maxLatencyMs = sorted.lastOrNull() ?: 0.0,
            chunkStats = chunkStats.toList()
        )
    }

    /** Clear all stored metrics */
    fun reset() {
        latencies.clear()
        chunkStats.clear()
        totalSamples = 0
    }

    private fun getBatteryCurrentUa(): Float {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val currentNow = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        return kotlin.math.abs(currentNow.toFloat()) // treat charging as positive
    }

    /** Returns approximate CPU usage (%) since last call */
    private fun getCpuUsagePercent(): Float {
        try {
            val reader1 = RandomAccessFile("/proc/stat", "r")
            val line1 = reader1.readLine()
            reader1.close()

            val toks1 = line1.split("\\s+".toRegex())
            val idle1 = toks1[4].toLong()
            val total1 = toks1.drop(1).map { it.toLong() }.sum()

            // Wait a tiny bit (200ms)
            Thread.sleep(200)

            val reader2 = RandomAccessFile("/proc/stat", "r")
            val line2 = reader2.readLine()
            reader2.close()

            val toks2 = line2.split("\\s+".toRegex())
            val idle2 = toks2[4].toLong()
            val total2 = toks2.drop(1).map { it.toLong() }.sum()

            val totalDiff = total2 - total1
            val idleDiff = idle2 - idle1

            if (totalDiff == 0L) return 0f
            return ((totalDiff - idleDiff).toFloat() / totalDiff.toFloat()) * 100f

        } catch (e: Exception) {
            e.printStackTrace()
            return 0f
        }
    }
}