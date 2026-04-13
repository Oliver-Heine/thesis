package com.thesis.qrquishing.model.external

//import com.thesis.qrquishing.view.MainActivity.Companion.BACKEND_URL
import com.thesis.qrquishing.model.dto.ModelResult
import com.thesis.qrquishing.model.dto.Verdict

class BackendService {

    fun validate(url: String): ModelResult {
        return ModelResult("Not setup yet", Verdict.UNCERTAIN, 0.42f)
//        return try {
//            val body = JSONObject().put("url", url).toString().toRequestBody("application/json".toMediaType())
//            val request = Request.Builder().url(BACKEND_URL).post(body).build()
//            httpClient.newCall(request).execute().use { response ->
//                if (response.isSuccessful) response.body?.string()?.let { JSONObject(it) } else null
//            }
//        } catch (e: Exception) {
//            null
//        }
    }
}