package com.thesis.qrquishing.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.thesis.qrquishing.model.external.BackendService
import com.thesis.qrquishing.model.ai.TFLiteClassifier
import com.thesis.qrquishing.model.dto.LocalResult
import com.thesis.qrquishing.model.dto.ModelResults
import com.thesis.qrquishing.model.dto.Verdict
import com.thesis.qrquishing.utils.Event
import com.thesis.qrquishing.utils.Settings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainViewModel(
    private val classifier: TFLiteClassifier
) : ViewModel() {

    private val _result = MutableLiveData<ModelResults>()
    val result: LiveData<ModelResults> = _result

    private val _warningEvent = MutableLiveData<Event<ModelResults>>()
    val warningEvent: LiveData<Event<ModelResults>> = _warningEvent
    val userAllowedOffDevice: Boolean = true
    val externalBackendService: BackendService = BackendService()

    fun onQrScanned(rawUrl: String) {
        val url = rawUrl.trim()
        if (url.isBlank()) return

        analyzeUrl(url)
    }

    private fun analyzeUrl(url: String) {
        viewModelScope.launch {
            val (verdict, confidence) = withContext(Dispatchers.Default) {
                classifier.classify(url)
            }

            var modelResults = ModelResults()

            modelResults = modelResults.copy(localResult = LocalResult(url, verdict, confidence))

            if (Settings.ALWAYS_USE_BACKEND ||
                (verdict == Verdict.UNCERTAIN && Settings.backendEnabled)) {
                modelResults = modelResults.copy(
                    backendResult = withContext(Dispatchers.IO) {externalBackendService.validate(url)})
            }
            _result.value = modelResults
        }
    }
}