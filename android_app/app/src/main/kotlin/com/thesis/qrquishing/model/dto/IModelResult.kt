package com.thesis.qrquishing.model.dto

interface IModelResult {
    val url: String
    val verdict: Verdict
    val confidence: Float
}