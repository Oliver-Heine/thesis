package com.thesis.qrquishing.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.thesis.qrquishing.ai.TFLiteClassifier
import com.thesis.qrquishing.domain.ModelResult
import com.thesis.qrquishing.utils.Event
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainViewModel(
    private val classifier: TFLiteClassifier
) : ViewModel() {

    private val _result = MutableLiveData<ModelResult>()
    val result: LiveData<ModelResult> = _result

    private val _warningEvent = MutableLiveData<Event<ModelResult>>()
    val warningEvent: LiveData<Event<ModelResult>> = _warningEvent

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

            val modelResult = ModelResult(url, verdict, confidence)

            _result.value = modelResult

            if (verdict.shouldWarn()) {
                _warningEvent.value = Event(modelResult)
            }
        }
    }
}