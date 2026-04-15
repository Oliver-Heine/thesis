package com.thesis.qrquishing.model.external
import com.thesis.qrquishing.model.dto.BackendResult
import com.thesis.qrquishing.model.dto.Verdict
import com.thesis.qrquishing.utils.Settings
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import okhttp3.Request
import okhttp3.OkHttpClient

class BackendService {
    private val httpClient = OkHttpClient()

    fun validate(url: String): BackendResult {

        return try {
            var httpUrl = url
            if (!url.contains("http")) {
                httpUrl = "http://$url"
            }

            val body = JSONObject().put("url", httpUrl).toString()
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url(String.format("%s/analyze", Settings.BACKEND_URL))
                .post(body)
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val json = response.body?.string()?.let { JSONObject(it) }
                    BackendResult(
                        url = json?.optString("url") ?: "No url",
                        verdict = Verdict.valueOf(json?.optString("prediction") ?: "UNCERTAIN"),
                        confidence = json?.optDouble("confidence", 0.42)?.toFloat() ?: 0.42f
                    )
                } else {
                    BackendResult("Request Failed", Verdict.UNCERTAIN, 0.0f)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            BackendResult("Error: ${e.message}", Verdict.UNCERTAIN, 0.0f)
        }
    }
}