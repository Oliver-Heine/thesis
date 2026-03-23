package com.thesis.qrquishing.model.dto

import com.thesis.qrquishing.R

enum class Verdict(val displayResId: Int, val colorResId: Int) {
    MALICIOUS(R.string.verdict_malicious_display_name, android.R.color.holo_red_dark),
    BENIGN(R.string.verdict_benign_display_name, android.R.color.holo_green_dark),
    UNCERTAIN(R.string.verdict_uncertain_display_name, android.R.color.holo_orange_dark);

    fun shouldWarn(): Boolean {
        return this == MALICIOUS || this == UNCERTAIN
    }
}