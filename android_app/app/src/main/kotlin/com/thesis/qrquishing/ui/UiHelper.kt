package com.thesis.qrquishing.ui

import android.app.AlertDialog
import android.content.Context
import android.view.View
import androidx.core.content.ContextCompat
import com.thesis.qrquishing.databinding.ActivityMainBinding
import com.thesis.qrquishing.domain.Verdict

class UiHelper(
    private val context: Context,
    private val binding: ActivityMainBinding
) {

    fun init() {
        binding.apply {
            cardResult.visibility = View.GONE
            bannerWarning.visibility = View.GONE
        }
    }

    fun updateResult(url: String, verdict: Verdict, confidence: Float) {
        binding.apply {
            cardResult.visibility = View.VISIBLE
            tvUrl.text = url
            tvVerdict.text = context.getString(verdict.displayResId)
            tvVerdict.setTextColor(ContextCompat.getColor(context, verdict.colorResId))

            val pct = confidence.toPercentage()
            progressConfidence.progress = pct
            tvConfidence.text = "Confidence: $pct %"

            bannerWarning.visibility = if (verdict.shouldWarn()) View.VISIBLE else View.GONE
        }
    }

    fun showWarningDialog(url: String, verdict: Verdict, confidence: Float, extra: String? = null) {
        val message = buildString {
            append("URL: $url\n\n")
            append("Verdict: $verdict\n")
            append("Confidence: ${confidence.toPercentage()}%\n\n")
            append("Do NOT open this URL.")
        }
        showDialog("⚠ Warning", message)
    }

    fun showPermissionDeniedDialog() {
        showDialog("Permission Denied", "Camera access is required.")
    }

    private fun showDialog(title: String, message: String) {
        AlertDialog.Builder(context)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun Float.toPercentage(): Int = (this * 100).toInt()
}