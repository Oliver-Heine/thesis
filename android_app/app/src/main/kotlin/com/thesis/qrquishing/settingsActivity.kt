package com.thesis.qrquishing

import android.content.Intent
import android.os.Bundle
import android.os.PersistableBundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.slider.Slider
import kotlinx.coroutines.cancel

class settingsActivity : AppCompatActivity()  {

    private var CONFIDENCE_THRESHOLD = 0.80f
    private lateinit var btnSwitchToCam: FloatingActionButton
    private lateinit var confidenceCutoff: Slider
    private lateinit var tvSliderValue: android.widget.TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings)
        btnSwitchToCam = findViewById(R.id.btnSwitchToCam)
        tvSliderValue = findViewById(R.id.sliderValue)

        confidenceCutoff = findViewById<Slider>(R.id.sliderSensitivity)
        tvSliderValue.text = confidenceCutoff.value.toString()

        confidenceCutoff.addOnChangeListener { _, value, _ ->
            CONFIDENCE_THRESHOLD = value / 100f
            tvSliderValue.text = value.toString()
        }

        btnSwitchToCam.setOnClickListener {
            var intent = Intent(this, DebugActivity::class.java)
            intent.putExtra("CONFIDENCE_THRESHOLD", CONFIDENCE_THRESHOLD)
            startActivity(intent)
        }

    }

    override fun onDestroy() {
        super.onDestroy()
    }
}