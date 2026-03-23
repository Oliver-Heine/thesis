package com.thesis.qrquishing.benchmarkrunner.dto

data class ChunkStats(
    val chunkIndex: Int,
    val avgLatencyMs: Double,
    val sampleCount: Int,
    val avgCpuPercent: Float = 0f,
    val avgBatteryCurrentUa: Float = 0f
)
