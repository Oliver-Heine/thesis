package com.thesis.qrquishing.benchmarkrunner.dto

data class InferenceMetrics(
    val totalSamples: Int,
    val avgLatencyMs: Double,
    val p50LatencyMs: Double,
    val p95LatencyMs: Double,
    val p99LatencyMs: Double,
    val minLatencyMs: Double,
    val maxLatencyMs: Double,
    val chunkStats: List<ChunkStats>,
)