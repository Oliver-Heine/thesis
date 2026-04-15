package com.thesis.qrquishing.utils

import com.thesis.qrquishing.databinding.SettingsBinding

class SettingsUiHelper (
    private val binding: SettingsBinding
) {

    fun init() {
        binding.sliderConfidence.value = Settings.CONFIDENCE_THRESHOLD
        binding.sliderValue.text = (Settings.CONFIDENCE_THRESHOLD).toString()
        binding.backendCheckBox.isChecked = Settings.backendEnabled
        binding.backendAlwaysCheckBox.isChecked = Settings.ALWAYS_USE_BACKEND
        binding.urlSourceText.setText(Settings.BACKEND_URL)
    }

    fun updateConfidenceCutoff(value: Float) {
        binding.sliderValue.text = value.toString()
        Settings.CONFIDENCE_THRESHOLD = value
    }

}