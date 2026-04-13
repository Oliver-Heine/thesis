package com.thesis.qrquishing.utils

import com.thesis.qrquishing.databinding.SettingsBinding

class SettingsUiHelper (
    private val binding: SettingsBinding
) {

    fun init() {
        binding.sliderConfidence.value = Settings.CONFIDENCE_THRESHOLD
        binding.sliderValue.text = (Settings.CONFIDENCE_THRESHOLD).toString()
    }

    fun updateConfidenceCutoff(value: Float) {
        binding.sliderValue.text = value.toString()
        Settings.CONFIDENCE_THRESHOLD = value
    }

}