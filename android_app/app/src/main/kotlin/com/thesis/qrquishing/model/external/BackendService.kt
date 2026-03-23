package com.thesis.qrquishing.data

//import com.thesis.qrquishing.view.MainActivity.Companion.BACKEND_URL
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class BackendService {

    fun validate(url: String): Boolean {
        return true
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