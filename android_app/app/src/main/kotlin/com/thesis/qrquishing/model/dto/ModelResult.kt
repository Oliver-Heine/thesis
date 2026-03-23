package com.thesis.qrquishing.model.dto

data class ModelResult(
    val url: String,
    val verdict: Verdict,
    val confidence: Float
)