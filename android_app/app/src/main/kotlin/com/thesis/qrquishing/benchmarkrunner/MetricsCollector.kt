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
    fun addChunk(latenciesChunk: List<Double>, chunkIndex: Int, chunkTime: Double) {
        if (latenciesChunk.isEmpty()) return

        latencies.addAll(latenciesChunk)
        totalSamples += latenciesChunk.size

        val avgLatency = latenciesChunk.average()
        val avgBatteryCurrent = getBatteryDischargeUa()

        chunkStats.add(
            ChunkStats(
                chunkIndex = chunkIndex,
                avgLatencyMs = avgLatency,
                sampleCount = latenciesChunk.size,
                chunkRunTime = chunkTime,
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
            val index = ((p * (sorted.size - 1))).toInt()
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

    private fun getBatteryDischargeUa(): Float {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val currentNow = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

        return currentNow.toFloat()
    }
}