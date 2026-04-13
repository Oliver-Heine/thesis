package com.thesis.qrquishing.view

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.slider.Slider
import com.thesis.qrquishing.R
import com.thesis.qrquishing.databinding.SettingsBinding
import com.thesis.qrquishing.utils.Settings
import com.thesis.qrquishing.utils.SettingsUiHelper

class SettingsActivity : AppCompatActivity()  {
    private lateinit var binding: SettingsBinding
    private lateinit var uiHelper: SettingsUiHelper



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupBinding()
        setListeners()
        setupUi()
    }


    private fun setupUi() {
        uiHelper = SettingsUiHelper(binding)
        uiHelper.init()
    }

    private fun setupBinding() {
        binding = SettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)
    }

    private fun setListeners() {
        binding.btnSwitchToCam.setOnClickListener { finish() }

        binding.backendCheckBox.setOnCheckedChangeListener { _, isChecked ->
            Settings.backendEnabled = isChecked
        }

        binding.sliderConfidence.addOnChangeListener { _, value, _ ->
            uiHelper.updateConfidenceCutoff(value)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}