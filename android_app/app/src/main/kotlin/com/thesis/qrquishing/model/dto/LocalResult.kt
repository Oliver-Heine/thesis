package com.thesis.qrquishing.model.dto

data class LocalResult(
    override val url: String,
    override val verdict: Verdict,
    override val confidence: Float
) : IModelResult