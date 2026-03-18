package com.thesis.qrquishing.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.thesis.qrquishing.ai.TFLiteClassifier

class MainViewModelFactory(
    private val classifier: TFLiteClassifier
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            return MainViewModel(classifier) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}