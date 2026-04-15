package com.thesis.qrquishing.utils

import android.app.AlertDialog
import android.content.Context
import android.view.View
import androidx.core.content.ContextCompat
import com.thesis.qrquishing.databinding.ActivityMainBinding
import com.thesis.qrquishing.model.dto.IModelResult
import com.thesis.qrquishing.model.dto.ModelResults

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

    fun updateResult(modelResults: ModelResults) {
        binding.apply {
            val context = root.context

            // 1. Handle Local Result (Always show)
            modelResults.localResult?.let { local ->
                cardResult.visibility = View.VISIBLE
                tvUrl.text = local.url

                // Set local-specific views
                tvLocalVerdict.text = context.getString(local.verdict.displayResId)
                tvLocalVerdict.setTextColor(ContextCompat.getColor(context, local.verdict.colorResId))

                val localPct = local.confidence.toPercentage()
                tvLocalConfidence.text = "Local: $localPct%"

                // Global warning banner based on local (or both)
                bannerWarning.visibility = if (local.verdict.shouldWarn()) View.VISIBLE else View.GONE
            }

            // 2. Handle Backend Result (Side-by-side logic)
            if (modelResults.backendResult != null) {
                val backend = modelResults.backendResult!!
                containerBackend.visibility = View.VISIBLE

                tvBackendVerdict.text = context.getString(backend.verdict.displayResId)
                tvBackendVerdict.setTextColor(ContextCompat.getColor(context, backend.verdict.colorResId))

                val backendPct = backend.confidence.toPercentage()
                tvBackendConfidence.text = "Server: $backendPct%"

                // Optional: update banner if server finds something local missed
                if (backend.verdict.shouldWarn()) bannerWarning.visibility = View.VISIBLE

            } else {
                // Hide the server side if it hasn't loaded yet
                containerBackend.visibility = View.GONE
            }
        }
    }

    fun showWarningDialog(
        modelResults: ModelResults,
        extra: String? = null
    ) {

        val modelResult = determineModelResult(modelResults)

        val message = buildString {
            append("URL: ${modelResult.url}\n\n")
            append("Verdict: $modelResult.verdict\n")
            append("Confidence: ${modelResult.confidence.toPercentage()}%\n\n")
            append("Do NOT open this URL.")
        }
        showDialog("⚠ Warning", message)
    }

    fun determineModelResult(modelResults: ModelResults): IModelResult {
        return modelResults.backendResult ?: modelResults.localResult!!
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

}

fun Float.toPercentage(): Int {
    return (this * 100).toInt()
}