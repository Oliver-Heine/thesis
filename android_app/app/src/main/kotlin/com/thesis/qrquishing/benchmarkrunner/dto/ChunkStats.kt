package com.thesis.qrquishing.benchmarkrunner.dto

data class ChunkStats(
    val chunkIndex: Int,
    val avgLatencyMs: Double,
    val sampleCount: Int,
    val chunkRunTime: Double = 0.0,
)
