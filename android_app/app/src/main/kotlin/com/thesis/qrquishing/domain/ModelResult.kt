package com.thesis.qrquishing.domain

data class ModelResult(
    val url: String,
    val verdict: Verdict,
    val confidence: Float
)